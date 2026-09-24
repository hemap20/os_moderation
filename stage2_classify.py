"""Stage 2 — Gemini ground-truth classification.

Text-only: takes the Stage 1 transcript (never audio) plus prompt.py — the
SAME classification prompt that Stage 3 sends to Gemma with the raw audio
instead. Same policy text, same output schema; only the input modality
differs. All policy/classification logic lives in prompt.py, not here.

You run this yourself, interactively, after Stage 1 is complete for the
files you care about:

    python3 stage2_classify.py --dry-run
    python3 stage2_classify.py
    python3 stage2_classify.py --force
    python3 stage2_classify.py --limit 20
"""
import argparse
import json
import time
import traceback

import config
import dataset
import gemini_client
import prompt_loader
from pipeline_logging import StageLogger
from schemas import RawModelOutput, raw_model_output_json_schema, raw_to_ground_truth
import schemas as _ground_truth_schema_module

# Ground truth must ALWAYS be generated with the original schema, regardless
# of whatever experimental prompt version (e.g. prompt_v4.py) candidate
# models are being run against — this assertion catches it immediately if
# config.STAGE2_PROMPT_PATH is ever pointed at a prompt that prompt_loader's
# mapping resolves to a different schema module (see prompt_loader.
# schema_module_for_prompt), rather than silently generating ground truth
# with an experimental output schema.
assert prompt_loader.schema_module_for_prompt(config.STAGE2_PROMPT_PATH) is _ground_truth_schema_module, (
    f"config.STAGE2_PROMPT_PATH ({config.STAGE2_PROMPT_PATH}) maps to a non-ground-truth schema module — "
    "ground truth must always use schemas.py. Check prompt_loader._SCHEMA_MODULE_BY_PROMPT_NAME."
)

RAW_SCHEMA_STR = json.dumps(raw_model_output_json_schema())


def load_transcript_text(file_id: str) -> str:
    """Returns the timestamped, per-segment transcript text (NOT full_text,
    which is only the concatenated words with timestamps stripped) — Stage 2
    must see real per-segment timestamps so it can cite them accurately,
    rather than inventing its own timeline from unstamped running text."""
    p = config.TRANSCRIPTS_DIR / f"{file_id}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"No Stage 1 transcript for {file_id}. Run stage1_transcribe.py first."
        )
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        raise RuntimeError(f"Stage 1 transcript for {file_id} is not successful: {data.get('error')}")

    segments = data.get("segments", [])
    if not segments:
        return data.get("full_text", "")

    lines = [f"[{seg.get('t')}] {seg.get('text', '')}" for seg in segments]
    return "\n".join(lines)


def call_for_classification(client, file_id: str, logger: StageLogger) -> str:
    """Calls Gemini and returns the raw text response (retried). Kept
    separate from parsing so callers can persist the raw response for
    debugging BEFORE attempting to parse/validate it."""
    transcript_text = load_transcript_text(file_id)
    # prompt.py has no {transcript} placeholder (it's written for audio
    # input) — the transcript is supplied as a separate text part instead of
    # the audio file, in place of the file upload Stage 3 attaches.
    prompt_text = prompt_loader.render_prompt(
        config.STAGE2_PROMPT_PATH,
        json_schema_str=RAW_SCHEMA_STR,
    )
    transcript_part = (
        "[TRANSCRIPT OF THE CALL — analyse this text, which is the full "
        "transcription of the audio, in place of the audio itself]\n\n"
        f"{transcript_text}"
    )

    def do_call():
        return gemini_client.generate_text(
            client, config.GEMINI_MODEL, contents=[prompt_text, transcript_part],
            response_json_schema=raw_model_output_json_schema(),
        )

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"{file_id}: classification attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    return gemini_client.call_with_retries(do_call, on_retry=on_retry)


def parse_classification(file_id: str, raw_text: str):
    raw_output = RawModelOutput(**gemini_client.parse_json_lenient(raw_text))
    return raw_to_ground_truth(file_id, raw_output)


def already_done(file_id: str) -> bool:
    p = config.CLASSIFICATIONS_DIR / f"{file_id}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "success"
    except Exception:
        return False


def write_raw_response(file_id: str, raw_text: str):
    """Persisted immediately after the API call, before parsing/validation —
    a parse failure must never mean the raw response is lost."""
    config.RAW_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_RESPONSES_DIR / f"{file_id}_classification.json").write_text(raw_text)


def write_result(result):
    config.CLASSIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (config.CLASSIFICATIONS_DIR / f"{result.file_id}.json").write_text(result.model_dump_json(indent=2))


def write_error(file_id: str, error: str):
    from schemas import GroundTruthClassification

    config.CLASSIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    err = GroundTruthClassification(file_id=file_id, status="error", error=error)
    (config.CLASSIFICATIONS_DIR / f"{file_id}.json").write_text(err.model_dump_json(indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logger = StageLogger("stage2_classify")

    try:
        prompt_loader.load_raw_prompt(config.STAGE2_PROMPT_PATH)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    records = dataset.load_dataset()
    file_ids = sorted({r.file_id for r in records})

    if args.dry_run:
        file_ids = file_ids[: args.dry_run_limit]
        logger.info(f"DRY RUN: classifying {len(file_ids)} file(s), nothing will be written")
    else:
        if not args.force:
            before = len(file_ids)
            file_ids = [fid for fid in file_ids if not already_done(fid)]
            logger.info(f"Skipping {before - len(file_ids)} already-classified file(s) (use --force to redo)")
        if args.limit:
            file_ids = file_ids[: args.limit]

    client = gemini_client.get_client()
    n_success, n_error = 0, 0
    t0 = time.time()

    for i, file_id in enumerate(file_ids, 1):
        logger.info(f"[{i}/{len(file_ids)}] Classifying {file_id}")
        try:
            raw_text = call_for_classification(client, file_id, logger)
            if not args.dry_run:
                write_raw_response(file_id, raw_text)

            result = parse_classification(file_id, raw_text)
            n_success += 1
            if args.dry_run:
                print("\n" + "=" * 80)
                print(f"FILE: {file_id}")
                print("=" * 80)
                print(result.model_dump_json(indent=2))
            else:
                write_result(result)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            tb = traceback.format_exc()
            logger.error(f"{file_id}: FAILED — {exc}\n{tb}")
            if not args.dry_run:
                write_error(file_id, str(exc))

    elapsed = time.time() - t0
    logger.info(f"Stage 2 done in {elapsed:.1f}s — success={n_success} error={n_error}")
    if args.dry_run:
        logger.info("DRY RUN complete — no files were written. Review output above.")


if __name__ == "__main__":
    main()
