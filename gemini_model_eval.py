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
from schemas_gemma import GemmaChunkFlag, GemmaFileResult

# Fields tracked as "missing" per flag — mirrors gemma_local.py's
# _MISSING_TRACKED_FIELDS. Gemini's API has no logprobs (see module
# docstring), so this is the only per-field reliability signal available
# for it.
_MISSING_TRACKED_FIELDS = ["c", "speech_act", "quote_type", "violation"]

# Mutable module globals (not frozen constants) — reassigned by main() from
# --dataset-root via config.dataset_paths(), same pattern as gemma_local.py.
DATASET_DIR: Path = config.DOSTT_DIR
GEMINI_RESULTS_DIR: Path = config.PROJECT_ROOT / "gemini_results"
# Reassigned by main() from --prompt-path — see gemma_local.py's PROMPT_PATH
# for why this is separate from what generated ground truth.
PROMPT_PATH: Path = config.CLASSIFICATION_PROMPT_PATH
# Reassigned by main() alongside PROMPT_PATH — see gemma_local.py's
# equivalent global for why this must track the active prompt's schema.
RAW_SCHEMA_STR = json.dumps(prompt_loader.schema_module_for_prompt(PROMPT_PATH).raw_model_output_json_schema())


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


def parse_classification(raw_text: str) -> tuple:
    """Returns (flags, missing_field_counts). Schema-aware: builds
    RawModelOutputV4 (with the extra speech_act/quote_type/violation
    fields, all Optional) when PROMPT_PATH maps to schemas_v4, else the
    original RawModelOutput — same mapping gemma_local.py uses, so both
    runners can never disagree about which prompt uses which schema.
    Gemini has no logprobs (see module docstring), so the v4
    logprob_violation/p_violation_yes/p_violation_method/
    violation_token_topk fields always stay None here — only the
    categorical speech_act/quote_type/violation fields are populated."""
    schema_module = prompt_loader.schema_module_for_prompt(PROMPT_PATH)
    is_v4 = schema_module is not None and hasattr(schema_module, "RawModelOutputV4")
    parsed = gemini_client.parse_json_lenient(raw_text)
    raw_output = schema_module.RawModelOutputV4(**parsed) if is_v4 else schema_module.RawModelOutput(**parsed)

    missing_counts = {k: 0 for k in _MISSING_TRACKED_FIELDS}
    flags = []
    for f in raw_output.d:
        if f.confidence is None:
            missing_counts["c"] += 1
        speech_act = getattr(f, "speech_act", None)
        quote_type = getattr(f, "quote_type", None)
        violation = getattr(f, "violation", None)
        if is_v4:
            if speech_act is None:
                missing_counts["speech_act"] += 1
            if quote_type is None:
                missing_counts["quote_type"] += 1
            if violation is None:
                missing_counts["violation"] += 1
        flags.append(GemmaChunkFlag(
            model_category=f.flag, model_timestamp=f.timestamp, model_excerpt=f.excerpt,
            model_translation=f.translation, model_justification=f.justification,
            model_confidence=f.confidence,
            model_speech_act=speech_act, model_quote_type=quote_type, model_violation=violation,
        ))
    return flags, missing_counts


def process_file(client, model: str, record: dsv2.FileRecordV2, logger: StageLogger):
    raw_text = call_classification(client, model, record, logger)
    flags, missing_counts = parse_classification(raw_text)
    result = GemmaFileResult(
        file_id=record.file_id, model=model, thinking=False, chunk_seconds=0.0,
        flags=flags, chunks_total=1, chunks_failed=0, status="success",
        missing_field_counts=missing_counts,
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

    global DATASET_DIR, GEMINI_RESULTS_DIR, PROMPT_PATH, RAW_SCHEMA_STR
    paths = config.dataset_paths(args.dataset_root, args.results_tag)
    DATASET_DIR = paths["dataset_dir"]
    GEMINI_RESULTS_DIR = paths["gemini_results_dir"]
    if args.prompt_path:
        p = Path(args.prompt_path)
        PROMPT_PATH = p if p.is_absolute() else config.PROJECT_ROOT / p
    RAW_SCHEMA_STR = json.dumps(prompt_loader.schema_module_for_prompt(PROMPT_PATH).raw_model_output_json_schema())

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
    total_missing_counts = {k: 0 for k in _MISSING_TRACKED_FIELDS}
    counts_lock = threading.Lock()
    t0 = time.time()

    def process_one(i, record):
        nonlocal n_success, n_error
        logger.info(f"[{i}/{len(records)}] {record.file_id}")
        try:
            result, raw_text = process_file(client, args.model, record, logger)
            with counts_lock:
                n_success += 1
                for key, n in result.missing_field_counts.items():
                    total_missing_counts[key] = total_missing_counts.get(key, 0) + n
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

    missing_summary = ", ".join(f"{k}={v}" for k, v in total_missing_counts.items())
    logger.info(f"Done in {time.time() - t0:.1f}s — success={n_success} error={n_error} — missing fields (count of flags lacking each): {missing_summary}")


if __name__ == "__main__":
    main()
