"""Stage 3 variant — Gemma 4 E2B/E4B TEXT classification, fed the
Indic-Conformer (or, with --transcript-source itc, Indic-Transcribe-Core) transcript instead of raw audio; --thinking enables reasoning mode. Originally non-thinking only, per your
request. Reuses gemma_local.py's model loading, JSON parsing, logprob/
entropy scoring, and flag-extraction machinery directly (all of that is
audio-agnostic already) — only the input construction and per-file
orchestration differ, since text needs no chunking (Gemma's context window
is enormous; unlike audio there's no 30s cap).

UNVERIFIED like gemma_local.py — no GPU in this environment.

Usage:
    python3 gemma_local_text.py --model e2b --dry-run --dry-run-limit 2
    python3 gemma_local_text.py --model e2b
    python3 gemma_local_text.py --model e4b --transcript-chunk-seconds 5
"""
import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Optional

import config
import dataset_v2 as dsv2
import prompt_loader
from pipeline_logging import StageLogger
from schemas_gemma import GemmaFileResult

from gemma_local import (
    MODEL_IDS,
    load_model,
    parse_and_score_flags,
    token_logprobs_and_entropy,
)

# Mutable module globals (mirrors gemma_local.py's pattern) — reassigned by
# main() from --dataset-root/--prompt-path, so a dev-set/prompt-experiment
# run reads/writes the right places without every function needing its own
# override plumbing.
DATASET_DIR: Path = config.DOSTT_DIR
GEMMA_RESULTS_DIR: Path = config.PROJECT_ROOT / "gemma_results"
PROMPT_PATH: Path = config.CLASSIFICATION_PROMPT_PATH
RAW_SCHEMA_STR: str = json.dumps(prompt_loader.schema_module_for_prompt(PROMPT_PATH).raw_model_output_json_schema())
# --transcript-source -> (ASR results dir, tag used in output dir names). Both
# ASR pipelines write the same per-file transcript JSON shape. Default
# "conformer" keeps every existing output path unchanged.
TRANSCRIPT_SOURCES = {"conformer": ("indic_conformer_results", "conformer"),
                      "itc": ("indic_transcribe_core_results", "itc")}
TRANSCRIPT_SOURCE: str = "conformer"
# --thinking: Gemma's reasoning mode. Same handling as gemma_local.classify_chunk
# (parse_response strips the reasoning before the JSON is parsed).
THINKING: bool = False


def output_dir(model_key: str, transcript_chunk_seconds: float) -> Path:
    tag = TRANSCRIPT_SOURCES[TRANSCRIPT_SOURCE][1]
    mode = "thinking" if THINKING else "nothinking"
    return GEMMA_RESULTS_DIR / f"{model_key}_{tag}{transcript_chunk_seconds:g}s_text_{mode}"


def load_conformer_transcript_lines(transcript_chunk_seconds: float, file_id: str) -> Optional[str]:
    """Same '[t] text' per-line format used to feed Gemini in stage2_classify.py
    — gives the model real per-line timestamps to cite, rather than a single
    unstamped blob of text."""
    p = (config.PROJECT_ROOT / TRANSCRIPT_SOURCES[TRANSCRIPT_SOURCE][0] / f"chunk_{transcript_chunk_seconds:g}s"
         / "transcripts" / f"{file_id}.json")
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    segments = data.get("segments", [])
    if not segments:
        return data.get("full_text", "")
    return "\n".join(f"[{s.get('t')}] {s.get('text', '')}" for s in segments)


def classify_text(model, processor, transcript_text: str, logger: StageLogger):
    import torch

    prompt_text = prompt_loader.render_prompt(PROMPT_PATH, json_schema_str=RAW_SCHEMA_STR)
    transcript_part = (
        "[TRANSCRIPT OF THE CALL — analyse this text, which is the full "
        "transcription of the audio, in place of the audio itself]\n\n"
        f"{transcript_text}"
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": transcript_part},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=THINKING,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            # Thinking needs room for the reasoning before the JSON answer, and a
            # whole transcript is one call (no 30 s chunks), so give it 2x.
            **inputs, max_new_tokens=8192 if THINKING else 4096, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    raw_response = processor.decode(generated_ids[input_len:], skip_special_tokens=False)
    if THINKING:
        prefix_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        parsed = processor.parse_response(raw_response, prefix=prefix_text)
        answer_text = parsed.get("content", raw_response)
    else:
        answer_text = processor.decode(generated_ids[input_len:], skip_special_tokens=True)
    token_infos = token_logprobs_and_entropy(model, processor, input_len, generated_ids, outputs.scores)

    return answer_text, raw_response, token_infos


def process_file(model, processor, record: dsv2.FileRecordV2, model_key: str, transcript_chunk_seconds: float, logger: StageLogger):
    transcript_text = load_conformer_transcript_lines(transcript_chunk_seconds, record.file_id)
    if transcript_text is None:
        return None, None

    answer_text, raw_response, token_infos = classify_text(model, processor, transcript_text, logger)
    flags, missing_field_counts = parse_and_score_flags(answer_text, token_infos, chunk_offset_sec=0.0, logger=logger)

    result = GemmaFileResult(
        file_id=record.file_id, model=model_key, thinking=THINKING,
        chunk_seconds=transcript_chunk_seconds,  # repurposed here: the Indic-Conformer chunk size the input transcript came from
        flags=flags, chunks_total=1, chunks_failed=0, status="success",
        missing_field_counts=missing_field_counts,
    )
    return result, raw_response


def already_done(model_key: str, transcript_chunk_seconds: float, file_id: str) -> bool:
    p = output_dir(model_key, transcript_chunk_seconds) / "results" / f"{file_id}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "success"
    except Exception:
        return False


def write_result(model_key: str, transcript_chunk_seconds: float, result: GemmaFileResult, raw_response: str):
    base = output_dir(model_key, transcript_chunk_seconds)
    (base / "results").mkdir(parents=True, exist_ok=True)
    (base / "raw_responses").mkdir(parents=True, exist_ok=True)
    (base / "results" / f"{result.file_id}.json").write_text(result.model_dump_json(indent=2))
    (base / "raw_responses" / f"{result.file_id}.json").write_text(raw_response)


def write_error(model_key: str, transcript_chunk_seconds: float, file_id: str, error: str):
    base = output_dir(model_key, transcript_chunk_seconds)
    (base / "results").mkdir(parents=True, exist_ok=True)
    err = GemmaFileResult(file_id=file_id, model=model_key, thinking=THINKING, chunk_seconds=transcript_chunk_seconds, flags=[], status="error", error=error)
    (base / "results" / f"{file_id}.json").write_text(err.model_dump_json(indent=2))


def free_device_memory():
    """Release the generation's cached GPU memory after every call. Without this
    the MPS allocator keeps each call's KV cache around, the process grows from
    ~21 GB to 24 GB+ on a 24 GB Mac within ~15 calls, and everything swaps."""
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def unique_records():
    seen = set()
    for r in dsv2.load_dataset_v2(DATASET_DIR):
        if r.file_id in seen:
            continue
        seen.add(r.file_id)
        yield r


def main():
    global DATASET_DIR, GEMMA_RESULTS_DIR, PROMPT_PATH, RAW_SCHEMA_STR, TRANSCRIPT_SOURCE, THINKING

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["e2b", "e4b"], required=True)
    parser.add_argument("--thinking", action="store_true", help="Enable Gemma's reasoning mode")
    parser.add_argument("--transcript-source", choices=sorted(TRANSCRIPT_SOURCES), default="conformer",
                         help="conformer = indic_conformer_results (default), itc = indic_transcribe_core_results")
    parser.add_argument("--transcript-chunk-seconds", type=float, default=5.0,
                         help="Which Indic-Conformer chunk-size sweep's transcripts to use as input (default: 5s, the best-performing size per the semantic-WER comparison)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-root", type=str, default=None,
                         help="Alternate dataset root (e.g. Dostt_dev) — see config.dataset_paths(). Only affects which FILES are scored; the Indic-Conformer transcripts themselves are always read from the full dataset's indic_conformer_results/ (keyed by file_id, generated once for all files including dev-set ones).")
    parser.add_argument("--results-tag", type=str, default=None,
                         help="Extra suffix for this run's output dir, to keep a prompt experiment separate from the baseline — see config.dataset_paths().")
    parser.add_argument("--prompt-path", type=str, default=str(config.CLASSIFICATION_PROMPT_PATH),
                         help="Prompt file to classify with (e.g. prompt_v5.py) — independent of ground truth's prompt.py.")
    args = parser.parse_args()
    TRANSCRIPT_SOURCE = args.transcript_source
    THINKING = args.thinking

    paths = config.dataset_paths(args.dataset_root, args.results_tag)
    DATASET_DIR = paths["dataset_dir"]
    GEMMA_RESULTS_DIR = paths["gemma_results_dir"]
    p = Path(args.prompt_path)
    PROMPT_PATH = p if p.is_absolute() else config.PROJECT_ROOT / p
    RAW_SCHEMA_STR = json.dumps(prompt_loader.schema_module_for_prompt(PROMPT_PATH).raw_model_output_json_schema())

    logger = StageLogger(f"gemma_local_text_{args.model}_{TRANSCRIPT_SOURCES[TRANSCRIPT_SOURCE][1]}{args.transcript_chunk_seconds:g}s"
                         f"{'_thinking' if THINKING else ''}")
    logger.info(f"Loading {MODEL_IDS[args.model]} (text-only, {'thinking' if THINKING else 'non-thinking'})...")
    model, processor = load_model(args.model)
    logger.info("Model loaded.")

    records = list(unique_records())
    if args.dry_run:
        records = records[: args.dry_run_limit]
        logger.info(f"DRY RUN: {len(records)} file(s)")
    else:
        if not args.force:
            before = len(records)
            records = [r for r in records if not already_done(args.model, args.transcript_chunk_seconds, r.file_id)]
            logger.info(f"Skipping {before - len(records)} already-done file(s)")
        if args.limit:
            records = records[: args.limit]

    n_success, n_error, n_skipped = 0, 0, 0
    t0 = time.time()

    for i, record in enumerate(records, 1):
        if i > 1:
            free_device_memory()
        logger.info(f"[{i}/{len(records)}] {record.file_id}")
        try:
            result, raw_response = process_file(model, processor, record, args.model, args.transcript_chunk_seconds, logger)
            if result is None:
                logger.warn(f"{record.file_id}: no {TRANSCRIPT_SOURCE} transcript at chunk_seconds={args.transcript_chunk_seconds}, skipping")
                n_skipped += 1
                continue
            n_success += 1
            if args.dry_run:
                print(result.model_dump_json(indent=2))
            else:
                write_result(args.model, args.transcript_chunk_seconds, result, raw_response)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            logger.error(f"{record.file_id}: FAILED — {exc}\n{traceback.format_exc()}")
            if not args.dry_run:
                write_error(args.model, args.transcript_chunk_seconds, record.file_id, str(exc))

    logger.info(f"Done in {time.time() - t0:.1f}s — success={n_success} error={n_error} skipped={n_skipped}")


if __name__ == "__main__":
    main()
