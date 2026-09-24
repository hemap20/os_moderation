"""Stage 1+2 combined pipeline for the new Dostt dataset layout.

For every file:
  1. Transcribe (Stage 1 — same instruction/logic as before, no policy awareness).
  2. Classify against prompt.py (Stage 2 — same shared prompt/schema as before,
     text-only from the transcript just produced).
  3. Determine the file's ACTUAL bucket based on the classification result:
       - TP folder file, no confirmed matching-category violation -> flagged
         for review (review_needed/tp_mismatches.csv). Bucket is NOT changed
         automatically.
       - FP folder file, a violation IS confirmed -> flagged for review
         (review_needed/fp_mismatches.csv). Bucket is NOT changed automatically.
       - FN folder file (no pre-assigned category), a violation IS confirmed
         -> confirmed genuine FN, category = the classified category.
       - FN folder file, NO violation found -> reassigned (logically only,
         never a physical file move) to the TN pool, tagged with whichever
         category the content is topically closest to via a small,
         non-policy heuristic call, and always marked needs_review=True
         (recorded in manifest/fn_reassignment_manifest.csv).

Writes ONE combined JSON per file to {folder}/transcripts/{file_id}.json
(transcript, then classification, then determination) and raw API
responses to {folder}/raw_responses/{file_id}_{transcription,classification}.json.

You run this yourself, interactively:

    python3 stage12_v2.py --dry-run
    python3 stage12_v2.py
    python3 stage12_v2.py --force
    python3 stage12_v2.py --limit 20
"""
import argparse
import csv
import json
import time
import traceback

import config
import dataset_v2 as dsv2
import gemini_client
import prompt_loader
from pipeline_logging import StageLogger
from schemas import GroundTruthFlag, RawModelOutput, raw_model_output_json_schema, raw_to_ground_truth
import schemas as _ground_truth_schema_module
from schemas_v2 import ClassificationBlock, CombinedResult, Determination, TranscriptBlock
from stage1_transcribe import (
    TRANSCRIPTION_INSTRUCTION,
    call_for_transcription,
    get_audio_duration_sec,
    sanitize_segments,
    _parse_mmss_to_sec,
)

# See stage2_classify.py's identical assertion — ground truth must always
# use schemas.py regardless of what experimental prompt candidate models use.
assert prompt_loader.schema_module_for_prompt(config.STAGE2_PROMPT_PATH) is _ground_truth_schema_module, (
    f"config.STAGE2_PROMPT_PATH ({config.STAGE2_PROMPT_PATH}) maps to a non-ground-truth schema module — "
    "ground truth must always use schemas.py. Check prompt_loader._SCHEMA_MODULE_BY_PROMPT_NAME."
)

RAW_SCHEMA_STR = json.dumps(raw_model_output_json_schema())

TOPIC_TAG_INSTRUCTION = """
You are given a transcript of a phone/video call on a social app. No policy
violation was found in this transcript. For dataset bookkeeping purposes
ONLY (this is not a policy decision), pick which ONE of the following three
topics this conversation is MOST topically related to or adjacent to, even
though nothing here violates any policy:
- PlatformMove: conversations that touch on contact info, external apps,
  calls, phone/social numbers (even if never actually shared)
- SuspiciousActivity: conversations that touch on money, payment, rates,
  transactions (even if entirely legitimate)
- Explicit-Flirting: conversations that touch on romance, compliments, or
  relationships (even if mild and entirely appropriate)
If none stands out clearly, pick whichever topic came up most often or most
saliently. Return ONE raw, minified JSON object:
{"category": "PlatformMove" | "SuspiciousActivity" | "Explicit-Flirting", "rationale": "one sentence"}
Your entire response must be only the minified JSON object and nothing else.
""".strip()


# ---------------------------------------------------------------------------
# Stage 1 (reuses stage1_transcribe's call — FileRecordV2 is duck-type
# compatible: it has .path and .file_id, which is all that code needs).
# Call and parse are kept separate so the caller can persist the raw
# response BEFORE parsing — a parse failure must never mean the raw
# response is lost.
# ---------------------------------------------------------------------------
def call_transcription(client, record: dsv2.FileRecordV2, logger: StageLogger) -> str:
    return call_for_transcription(client, record, logger)


def parse_transcription(record: dsv2.FileRecordV2, raw_text: str, logger: StageLogger) -> TranscriptBlock:
    parsed = gemini_client.parse_json_lenient(gemini_client.repair_transcription_artifacts(raw_text))
    segments = sanitize_segments(parsed.get("segments", []), record.file_id, logger)
    # Derived, not requested from the model — see stage1_transcribe.py.
    full_text = " ".join(seg.get("text", "") for seg in segments)

    duration = get_audio_duration_sec(record.path)
    transcript_end = None
    for seg in reversed(segments):
        t_sec = _parse_mmss_to_sec(seg.get("t", ""))
        if t_sec is not None:
            transcript_end = t_sec
            break

    incomplete = False
    gap = None
    if duration is not None and transcript_end is not None:
        gap = duration - transcript_end
        threshold = max(
            config.INCOMPLETE_TRANSCRIPT_MIN_GAP_SEC,
            config.INCOMPLETE_TRANSCRIPT_GAP_FRACTION * duration,
        )
        if gap > threshold:
            incomplete = True
            logger.warn(
                f"{record.file_id}: INCOMPLETE_TRANSCRIPT — audio={duration:.1f}s, "
                f"transcript ends at {transcript_end:.1f}s, gap={gap:.1f}s (threshold={threshold:.1f}s)"
            )
    elif duration is None:
        logger.warn(f"{record.file_id}: could not read audio duration; skipping completeness check")

    return TranscriptBlock(
        segments=segments,
        full_text=full_text,
        audio_duration_sec=duration,
        transcript_end_sec=transcript_end,
        gap_sec=gap,
        incomplete_transcript=incomplete,
        status="success",
    )


def _transcript_lines(segments) -> str:
    return "\n".join(f"[{s.t}] {s.text}" for s in segments)


# ---------------------------------------------------------------------------
# Stage 2 — same shared prompt.py, fed the transcript we just produced. Call
# and parse kept separate for the same raw-response-persistence reason as
# Stage 1 above.
# ---------------------------------------------------------------------------
def call_classification(client, file_id: str, transcript_block: TranscriptBlock, logger: StageLogger) -> str:
    prompt_text = prompt_loader.render_prompt(
        config.STAGE2_PROMPT_PATH, json_schema_str=RAW_SCHEMA_STR
    )
    transcript_part = (
        "[TRANSCRIPT OF THE CALL — analyse this text, which is the full "
        "transcription of the audio, in place of the audio itself]\n\n"
        f"{_transcript_lines(transcript_block.segments)}"
    )

    def do_call():
        return gemini_client.generate_text(
            client, config.GEMINI_MODEL, contents=[prompt_text, transcript_part],
            response_json_schema=raw_model_output_json_schema(),
        )

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"{file_id}: classification attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    return gemini_client.call_with_retries(do_call, on_retry=on_retry)


def parse_classification(file_id: str, raw_text: str) -> ClassificationBlock:
    raw_output = RawModelOutput(**gemini_client.parse_json_lenient(raw_text))
    gt = raw_to_ground_truth(file_id, raw_output)
    return ClassificationBlock(ground_truth_flags=gt.ground_truth_flags, status="success")


# ---------------------------------------------------------------------------
# Topic tagging — only called for FN files with no confirmed violation, to
# give the TN reassignment a category. NOT a policy decision (no violation
# was found either way); always flagged needs_review.
# ---------------------------------------------------------------------------
TOPIC_TAG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["PlatformMove", "SuspiciousActivity", "Explicit-Flirting"]},
        "rationale": {"type": "string"},
    },
    "required": ["category", "rationale"],
}


def tag_topic_for_tn(client, transcript_block: TranscriptBlock, logger: StageLogger):
    transcript_part = f"\n\n{_transcript_lines(transcript_block.segments)}"

    def do_call():
        return gemini_client.generate_text(
            client, config.GEMINI_MODEL, contents=[TOPIC_TAG_INSTRUCTION, transcript_part],
            response_json_schema=TOPIC_TAG_JSON_SCHEMA,
        )

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"topic-tag attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
    parsed = gemini_client.parse_json_lenient(raw_text)
    return parsed.get("category"), parsed.get("rationale", "")


# ---------------------------------------------------------------------------
# Determination
# ---------------------------------------------------------------------------
def determine(
    client,
    record: dsv2.FileRecordV2,
    flags: list,
    transcript_block: TranscriptBlock,
    logger: StageLogger,
) -> Determination:
    if record.original_bucket == "TP":
        expected_cat = config.CATEGORY_LABELS[record.category]
        matched = [f for f in flags if f.ground_truth_category == expected_cat]
        if matched:
            return Determination(final_bucket="TP", final_category=expected_cat, needs_review=False)
        if flags:
            found = sorted({f.ground_truth_category for f in flags})
            note = f"Classifier found different categor(y/ies) {found}; folder assumed {expected_cat}."
        else:
            note = "NO VIOLATION FOUND — folder assumed TP, please replace this file."
        return Determination(
            final_bucket="TP", final_category=expected_cat, needs_review=True, review_reason=note
        )

    if record.original_bucket == "FP":
        expected_cat = config.CATEGORY_LABELS[record.category]
        if not flags:
            return Determination(final_bucket="FP", final_category=expected_cat, needs_review=False)
        found = sorted({f.ground_truth_category for f in flags})
        note = (
            f"FP file has confirmed violation(s): {found} — the original flag may have been "
            f"correct after all, or a different real violation exists."
        )
        return Determination(
            final_bucket="FP", final_category=expected_cat, needs_review=True, review_reason=note
        )

    # original_bucket == "FN"
    if flags:
        distinct = sorted({f.ground_truth_category for f in flags})
        final_cat = flags[0].ground_truth_category
        note = None
        needs_review = False
        if len(distinct) > 1:
            note = f"Multiple distinct violation categories found: {distinct}; final_category set to first ({final_cat})."
            needs_review = True
        return Determination(
            final_bucket="FN",
            final_category=final_cat,
            reassigned=False,
            reassignment_note=note,
            needs_review=needs_review,
            review_reason=note,
        )

    # No violation found in an FN file -> reassign to TN pool.
    suggested_cat, rationale = tag_topic_for_tn(client, transcript_block, logger)
    note = f"No violation found; reassigned to TN pool, tagged as {suggested_cat} ({rationale}) — needs review."
    logger.warn(f"{record.file_id}: FN->TN reassignment — {note}")
    return Determination(
        final_bucket="TN",
        final_category=suggested_cat,
        reassigned=True,
        reassignment_note=note,
        needs_review=True,
        review_reason="FN→TN category tag is a judgment call, not rule-based.",
    )


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------
def process_one(client, record: dsv2.FileRecordV2, logger: StageLogger, dry_run: bool = False) -> CombinedResult:
    raw_transcription = call_transcription(client, record, logger)
    if not dry_run:
        write_raw(record, "transcription", raw_transcription)
    transcript_block = parse_transcription(record, raw_transcription, logger)

    if transcript_block.status != "success":
        return CombinedResult(
            file_id=record.file_id,
            source_path=str(record.path),
            language=record.language,
            original_bucket=record.original_bucket,
            original_category=config.CATEGORY_LABELS.get(record.category) if record.category else None,
            transcript=transcript_block,
            classification=ClassificationBlock(status="error", error="skipped — transcript failed"),
            status="error",
            error="transcript failed",
        )

    raw_classification = call_classification(client, record.file_id, transcript_block, logger)
    if not dry_run:
        write_raw(record, "classification", raw_classification)
    classification_block = parse_classification(record.file_id, raw_classification)

    determination = determine(client, record, classification_block.ground_truth_flags, transcript_block, logger)

    return CombinedResult(
        file_id=record.file_id,
        source_path=str(record.path),
        language=record.language,
        original_bucket=record.original_bucket,
        original_category=config.CATEGORY_LABELS.get(record.category) if record.category else None,
        transcript=transcript_block,
        classification=classification_block,
        determination=determination,
        status="success",
    )


def write_raw(record: dsv2.FileRecordV2, kind: str, raw_text: str):
    record.raw_responses_dir.mkdir(parents=True, exist_ok=True)
    (record.raw_responses_dir / f"{record.file_id}_{kind}.json").write_text(raw_text)


def result_path(record: dsv2.FileRecordV2):
    return record.transcripts_dir / f"{record.file_id}.json"


def already_done(record: dsv2.FileRecordV2) -> bool:
    p = result_path(record)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "success"
    except Exception:
        return False


def write_result(record: dsv2.FileRecordV2, result: CombinedResult):
    record.transcripts_dir.mkdir(parents=True, exist_ok=True)
    result_path(record).write_text(result.model_dump_json(indent=2))


def write_error(record: dsv2.FileRecordV2, error: str):
    record.transcripts_dir.mkdir(parents=True, exist_ok=True)
    err = CombinedResult(
        file_id=record.file_id,
        source_path=str(record.path),
        language=record.language,
        original_bucket=record.original_bucket,
        original_category=config.CATEGORY_LABELS.get(record.category) if record.category else None,
        transcript=TranscriptBlock(status="error", error=error),
        classification=ClassificationBlock(status="error", error="skipped — upstream failure"),
        status="error",
        error=error,
    )
    result_path(record).write_text(err.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# End-of-run manifest / review CSVs — regenerated fresh from ALL completed
# results on disk each run, so resume/skip never produces duplicate rows.
# ---------------------------------------------------------------------------
def regenerate_reports(records):
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    fn_rows, tp_rows, fp_rows = [], [], []

    for record in records:
        p = result_path(record)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if data.get("status") != "success":
            continue
        det = data.get("determination")
        if not det:
            continue

        if record.original_bucket == "FN":
            fn_rows.append({
                "file_id": record.file_id,
                "source_path": str(record.path),
                "language": record.language,
                "original_bucket": "FN",
                "determined_bucket": det["final_bucket"],
                "determined_category": det.get("final_category") or "",
                "reassigned": det.get("reassigned", False),
                "needs_review": det.get("needs_review", False),
                "note": det.get("reassignment_note") or det.get("review_reason") or "",
            })
        elif record.original_bucket == "TP" and det.get("needs_review"):
            tp_rows.append({
                "file_path": str(record.path),
                "expected_category": data.get("original_category") or "",
                "actual_classification": json.dumps(data.get("classification", {}).get("ground_truth_flags", [])),
                "note": det.get("review_reason") or "",
            })
        elif record.original_bucket == "FP" and det.get("needs_review"):
            fp_rows.append({
                "file_path": str(record.path),
                "expected_status": "safe",
                "actual_classification": json.dumps(data.get("classification", {}).get("ground_truth_flags", [])),
                "note": det.get("review_reason") or "",
            })

    def write_csv(path, rows, fieldnames):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    write_csv(
        config.FN_MANIFEST_PATH, fn_rows,
        ["file_id", "source_path", "language", "original_bucket", "determined_bucket",
         "determined_category", "reassigned", "needs_review", "note"],
    )
    write_csv(
        config.TP_MISMATCH_CSV, tp_rows,
        ["file_path", "expected_category", "actual_classification", "note"],
    )
    write_csv(
        config.FP_MISMATCH_CSV, fp_rows,
        ["file_path", "expected_status", "actual_classification", "note"],
    )

    n_reassigned = sum(1 for r in fn_rows if r["reassigned"])
    return len(tp_rows), len(fp_rows), len(fn_rows), n_reassigned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logger = StageLogger("stage12_v2")

    try:
        prompt_loader.load_raw_prompt(config.STAGE2_PROMPT_PATH)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    all_records = dsv2.load_dataset_v2()
    logger.info(f"Loaded {len(all_records)} audio files from Dostt")

    if args.dry_run:
        # Sample across buckets so the dry-run demonstrates TP/FP/FN paths.
        by_bucket = {"TP": [], "FP": [], "FN": []}
        for r in all_records:
            by_bucket[r.original_bucket].append(r)
        records = []
        per_bucket = max(1, args.dry_run_limit // 3)
        for bucket in ("TP", "FP", "FN"):
            records.extend(by_bucket[bucket][:per_bucket])
        records = records[: args.dry_run_limit] if len(records) > args.dry_run_limit else records
        logger.info(f"DRY RUN: processing {len(records)} file(s) across TP/FP/FN, nothing will be written")
    else:
        records = all_records
        if not args.force:
            before = len(records)
            records = [r for r in records if not already_done(r)]
            logger.info(f"Skipping {before - len(records)} already-processed file(s) (use --force to redo)")
        if args.limit:
            records = records[: args.limit]

    client = gemini_client.get_client()
    n_success, n_error = 0, 0
    t0 = time.time()

    for i, record in enumerate(records, 1):
        logger.info(f"[{i}/{len(records)}] Processing {record.file_id} ({record.original_bucket}, {record.language})")
        try:
            result = process_one(client, record, logger, dry_run=args.dry_run)
            n_success += 1
            if args.dry_run:
                print("\n" + "=" * 80)
                print(f"FILE: {record.file_id}  [{record.original_bucket} / {record.language}]")
                print("=" * 80)
                print(result.model_dump_json(indent=2))
            else:
                write_result(record, result)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            tb = traceback.format_exc()
            logger.error(f"{record.file_id}: FAILED — {exc}\n{tb}")
            if not args.dry_run:
                write_error(record, str(exc))

    elapsed = time.time() - t0
    logger.info(f"Stage 1+2 (v2) done in {elapsed:.1f}s — success={n_success} error={n_error}")

    if args.dry_run:
        logger.info("DRY RUN complete — no files were written. Review output above.")
        return

    n_tp_mismatch, n_fp_mismatch, n_fn_total, n_reassigned = regenerate_reports(all_records)
    print("\n" + "=" * 80)
    print("END-OF-RUN SUMMARY")
    print("=" * 80)
    print(f"{n_tp_mismatch} TP files had no confirmed matching violation — review needed "
          f"(see {config.TP_MISMATCH_CSV})")
    print(f"{n_fp_mismatch} FP files had a confirmed violation — review needed "
          f"(see {config.FP_MISMATCH_CSV})")
    print(f"{n_fn_total} FN-origin files processed; {n_reassigned} reassigned to the TN pool "
          f"(see {config.FN_MANIFEST_PATH})")
    logger.info(
        f"Summary: tp_mismatches={n_tp_mismatch} fp_mismatches={n_fp_mismatch} "
        f"fn_total={n_fn_total} fn_reassigned_to_tn={n_reassigned}"
    )


if __name__ == "__main__":
    main()
