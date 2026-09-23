"""Evaluate an arbitrary Gemini model (native audio-in, single pass, same
shared prompt.py used for ground truth) against the full dataset — lets us
compare a different model/tier (e.g. gemini-3.5-flash-lite) against the
ground-truth model (gemini-3.1-flash-lite) the same way we've been
evaluating Gemma E2B/E4B/12B, but over the Gemini API instead of local
weights. No chunking needed — Gemini handles full-length audio natively
(unlike Gemma's 30s/call cap).

Note: logprobs are NOT available here — gemini-3.5-flash-lite (and likely
other Gemini API models) reject response_logprobs outright ("Logprobs is
not supported for this model"). So unlike the Gemma evaluations, there's no
logprob-derived confidence or token entropy here — only the model's own
self-reported confidence field, same as what's stored for ground truth.

Output uses the same model_-prefixed schema as the Gemma evaluations
(schemas_gemma.py) for consistent comparison in Stage 4, stored under
gemini_results/{model}/.

Usage:
    python3 gemini_model_eval.py --model gemini-3.5-flash-lite --dry-run --dry-run-limit 2
    python3 gemini_model_eval.py --model gemini-3.5-flash-lite
    python3 gemini_model_eval.py --model gemini-3.5-flash-lite --force --limit 20
"""
import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import config
import dataset_v2 as dsv2
import gemini_client
import prompt_loader
from pipeline_logging import StageLogger
from schemas import RawModelOutput, raw_model_output_json_schema
from schemas_gemma import GemmaChunkFlag, GemmaFileResult

RAW_SCHEMA_STR = json.dumps(raw_model_output_json_schema())


# Mutable module globals (not frozen constants) — reassigned by main() from
# --dataset-root via config.dataset_paths(), same pattern as gemma_local.py.
DATASET_DIR: Path = config.DOSTT_DIR
GEMINI_RESULTS_DIR: Path = config.PROJECT_ROOT / "gemini_results"
# Reassigned by main() from --prompt-path — see gemma_local.py's PROMPT_PATH
# for why this is separate from what generated ground truth.
PROMPT_PATH: Path = config.CLASSIFICATION_PROMPT_PATH


def output_dir(model: str) -> Path:
    safe_name = model.replace("/", "_")
    return GEMINI_RESULTS_DIR / safe_name


def call_classification(client, model: str, record: dsv2.FileRecordV2, logger: StageLogger) -> str:
    prompt_text = prompt_loader.render_prompt(PROMPT_PATH, json_schema_str=RAW_SCHEMA_STR)

    def do_call():
        uploaded = gemini_client.upload_audio(client, record.path)
        return gemini_client.generate_text(client, model, contents=[uploaded, prompt_text])

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"{record.file_id}: attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    return gemini_client.call_with_retries(do_call, on_retry=on_retry)


def parse_classification(raw_text: str) -> list:
    parsed = gemini_client.parse_json_lenient(raw_text)
    raw_output = RawModelOutput(**parsed)
    flags = []
    for f in raw_output.d:
        flags.append(GemmaChunkFlag(
            model_category=f.flag, model_timestamp=f.timestamp, model_excerpt="",
            model_translation=f.translation, model_justification="",
            model_confidence=f.confidence,
        ))
    return flags


def process_file(client, model: str, record: dsv2.FileRecordV2, logger: StageLogger):
    raw_text = call_classification(client, model, record, logger)
    flags = parse_classification(raw_text)
    result = GemmaFileResult(
        file_id=record.file_id, model=model, thinking=False, chunk_seconds=0.0,
        flags=flags, chunks_total=1, chunks_failed=0, status="success",
    )
    return result, raw_text


def already_done(model: str, file_id: str) -> bool:
    p = output_dir(model) / "results" / f"{file_id}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "success"
    except Exception:
        return False


def write_result(model: str, result: GemmaFileResult, raw_text: str):
    base = output_dir(model)
    (base / "results").mkdir(parents=True, exist_ok=True)
    (base / "raw_responses").mkdir(parents=True, exist_ok=True)
    (base / "results" / f"{result.file_id}.json").write_text(result.model_dump_json(indent=2))
    (base / "raw_responses" / f"{result.file_id}.json").write_text(raw_text)


def write_error(model: str, file_id: str, error: str):
    base = output_dir(model)
    (base / "results").mkdir(parents=True, exist_ok=True)
    err = GemmaFileResult(file_id=file_id, model=model, thinking=False, chunk_seconds=0.0, flags=[], status="error", error=error)
    (base / "results" / f"{file_id}.json").write_text(err.model_dump_json(indent=2))


def unique_records():
    seen = set()
    for r in dsv2.load_dataset_v2(DATASET_DIR):
        if r.file_id in seen:
            continue
        seen.add(r.file_id)
        yield r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. gemini-3.5-flash-lite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent Gemini calls (I/O-bound, threading is safe here)")
    parser.add_argument("--dataset-root", type=str, default=None,
                         help="Alternate dataset root, e.g. Dostt_dev — routes results to "
                              "gemini_results_<suffix>/ automatically; default uses the full Dostt/ dataset")
    parser.add_argument("--results-tag", type=str, default=None,
                         help="Extra results-directory suffix for a prompt experiment, e.g. "
                              "--results-tag promptA -> gemini_results_promptA/")
    parser.add_argument("--prompt-path", type=str, default=None,
                         help="Alternate prompt file for a prompt experiment, e.g. prompt_v2.py — "
                              "does NOT affect ground truth, which always used prompt.py")
    args = parser.parse_args()

    global DATASET_DIR, GEMINI_RESULTS_DIR, PROMPT_PATH
    paths = config.dataset_paths(args.dataset_root, args.results_tag)
    DATASET_DIR = paths["dataset_dir"]
    GEMINI_RESULTS_DIR = paths["gemini_results_dir"]
    if args.prompt_path:
        p = Path(args.prompt_path)
        PROMPT_PATH = p if p.is_absolute() else config.PROJECT_ROOT / p

    logger = StageLogger(f"gemini_model_eval_{args.model.replace('/', '_')}")

    try:
        prompt_loader.load_raw_prompt(PROMPT_PATH)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    records = list(unique_records())
    if args.dry_run:
        records = records[: args.dry_run_limit]
        logger.info(f"DRY RUN: {len(records)} file(s)")
    else:
        if not args.force:
            before = len(records)
            records = [r for r in records if not already_done(args.model, r.file_id)]
            logger.info(f"Skipping {before - len(records)} already-done file(s)")
        if args.limit:
            records = records[: args.limit]

    client = gemini_client.get_client()
    n_success, n_error = 0, 0
    counts_lock = threading.Lock()
    t0 = time.time()

    def process_one(i, record):
        nonlocal n_success, n_error
        logger.info(f"[{i}/{len(records)}] {record.file_id}")
        try:
            result, raw_text = process_file(client, args.model, record, logger)
            with counts_lock:
                n_success += 1
            if args.dry_run:
                print(result.model_dump_json(indent=2))
            else:
                write_result(args.model, result, raw_text)
        except Exception as exc:  # noqa: BLE001
            with counts_lock:
                n_error += 1
            logger.error(f"{record.file_id}: FAILED — {exc}\n{traceback.format_exc()}")
            if not args.dry_run:
                write_error(args.model, record.file_id, str(exc))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, i, record) for i, record in enumerate(records, 1)]
        for fut in as_completed(futures):
            fut.result()  # re-raise anything that escaped process_one's own try/except

    logger.info(f"Done in {time.time() - t0:.1f}s — success={n_success} error={n_error}")


if __name__ == "__main__":
    main()
