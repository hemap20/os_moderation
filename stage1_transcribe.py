"""Stage 1 — Gemini transcription.

Full, timestamped, native-language transcription of every audio file in the
dataset. Pure transcription only — no policy/category awareness, so (unlike
Stage 2/3) the instruction text here is not policy logic and is fine to keep
inline.

You run this yourself, interactively:

    python3 stage1_transcribe.py --dry-run              # 2-3 files, prints only
    python3 stage1_transcribe.py                          # full run, resumable
    python3 stage1_transcribe.py --force                  # redo everything
    python3 stage1_transcribe.py --limit 20                # cap file count
"""
import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Optional

from mutagen import File as MutagenFile

import config
import dataset
import gemini_client
from pipeline_logging import StageLogger
from schemas import Transcript

TRANSCRIPTION_INSTRUCTION = """
You are a precise audio transcription engine. Transcribe the ENTIRE attached
audio file from start to end, in the language(s) actually spoken (do not
translate). This is a two-person (sometimes one-person) phone/video call.

Rules:
- Cover the full duration of the audio, including trailing seconds. Do not
  stop early or summarize.
- Break the transcript into short lines, each tagged with a single
  timestamp "t" in MM:SS format marking when that line starts, relative to
  the start of the audio.
- Transcribe verbatim in the NATIVE SCRIPT of the language actually spoken
  (e.g. Devanagari for Hindi, Tamil script for Tamil, Telugu script for
  Telugu, Kannada script for Kannada, Malayalam script for Malayalam).
  Do NOT romanize/transliterate into Latin letters, even for code-switched
  English words embedded in the sentence — write the whole line in the
  native script, transliterating any English words into that script too.
- Do not classify, flag, or judge any content. Pure transcription only.
- If a line is inaudible/unintelligible, write "[inaudible]" for that
  line's text rather than guessing.

Return ONE raw, minified JSON object with this exact shape:
{"segments": [{"t": "MM:SS", "text": "..."}]}

Do not include any other top-level keys. Your entire response must be only
the minified JSON object and nothing else.
""".strip()

# Passed as response_json_schema so the API constrains decoding to this
# exact shape (grammar-level enforcement), rather than relying on the model
# to freely generate valid JSON — this is what eliminated most of the
# malformed-output failures observed with gemini-3.1-flash-lite in practice.
TRANSCRIPTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["t", "text"],
            },
        },
    },
    "required": ["segments"],
}


def get_audio_duration_sec(path: Path) -> Optional[float]:
    try:
        audio = MutagenFile(path)
        if audio is not None and audio.info is not None:
            return float(audio.info.length)
    except Exception:
        pass
    return None


def _parse_mmss_to_sec(ts: str) -> Optional[float]:
    try:
        parts = ts.strip().split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return None
    return None


def call_for_transcription(client, record: dataset.FileRecord, logger: StageLogger) -> str:
    """Calls Gemini and returns the raw text response (retried). Kept
    separate from parsing so callers can persist the raw response for
    debugging BEFORE attempting to parse/validate it — a parse failure must
    never mean the raw response is lost."""
    def do_call():
        uploaded = gemini_client.upload_audio(client, record.path)
        text = gemini_client.generate_text(
            client,
            config.GEMINI_MODEL,
            contents=[uploaded, TRANSCRIPTION_INSTRUCTION],
            response_json_schema=TRANSCRIPTION_JSON_SCHEMA,
        )
        return text

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(
            f"{record.file_id}: transcription attempt {attempt}/{max_retries} failed "
            f"({exc}); retrying in {delay:.1f}s"
        )

    return gemini_client.call_with_retries(do_call, on_retry=on_retry)


def sanitize_segments(raw_segments, file_id: str, logger: StageLogger) -> list:
    """Drop only the individual segments the model left malformed (missing
    't' or 'text', or wrong types), instead of letting one bad segment
    invalidate the whole file's transcript."""
    clean = []
    for i, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            logger.warn(f"{file_id}: dropping non-dict segment at index {i}: {seg!r}")
            continue
        t, text = seg.get("t"), seg.get("text")
        if not isinstance(t, str) or not isinstance(text, str):
            logger.warn(f"{file_id}: dropping malformed segment at index {i}: {seg!r}")
            continue
        clean.append({"t": t, "text": text})
    return clean


def parse_transcription(record: dataset.FileRecord, raw_text: str, logger: StageLogger) -> Transcript:
    parsed = gemini_client.parse_json_lenient(gemini_client.repair_transcription_artifacts(raw_text))

    segments = sanitize_segments(parsed.get("segments", []), record.file_id, logger)
    # Derived, not requested from the model — asking it to also emit a
    # separate full_text field meant regenerating every word a second time,
    # roughly doubling output size and truncation risk on long calls.
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
                f"transcript ends at {transcript_end:.1f}s, gap={gap:.1f}s "
                f"(threshold={threshold:.1f}s)"
            )
    elif duration is None:
        logger.warn(f"{record.file_id}: could not read audio duration; skipping completeness check")

    return Transcript(
        file_id=record.file_id,
        segments=segments,
        full_text=full_text,
        audio_duration_sec=duration,
        transcript_end_sec=transcript_end,
        gap_sec=gap,
        incomplete_transcript=incomplete,
        status="success",
    )


def already_done(file_id: str) -> bool:
    p = config.TRANSCRIPTS_DIR / f"{file_id}.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return data.get("status") == "success"
    except Exception:
        return False


def write_raw_response(file_id: str, raw_text: str):
    """Persisted immediately after the API call, before parsing/validation —
    a parse failure must never mean the raw response is lost."""
    config.RAW_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_RESPONSES_DIR / f"{file_id}_transcription.json").write_text(raw_text)


def write_result(transcript: Transcript):
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / f"{transcript.file_id}.json").write_text(
        transcript.model_dump_json(indent=2)
    )


def write_error(file_id: str, error: str):
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    err_transcript = Transcript(file_id=file_id, status="error", error=error)
    (config.TRANSCRIPTS_DIR / f"{file_id}.json").write_text(
        err_transcript.model_dump_json(indent=2)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Process 2-3 files, print only, write nothing")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true", help="Reprocess files that already succeeded")
    parser.add_argument("--limit", type=int, default=None, help="Cap total files processed this run")
    parser.add_argument("--max-retries", type=int, default=config.DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    logger = StageLogger("stage1_transcribe")
    records = dataset.load_dataset()
    logger.info(f"Loaded {len(records)} audio files from dataset")

    if args.dry_run:
        records = records[: args.dry_run_limit]
        logger.info(f"DRY RUN: processing {len(records)} file(s), nothing will be written")
    else:
        if not args.force:
            before = len(records)
            records = [r for r in records if not already_done(r.file_id)]
            logger.info(f"Skipping {before - len(records)} already-transcribed file(s) (use --force to redo)")
        if args.limit:
            records = records[: args.limit]

    client = gemini_client.get_client()

    n_success, n_error, n_incomplete = 0, 0, 0
    t0 = time.time()

    for i, record in enumerate(records, 1):
        logger.info(f"[{i}/{len(records)}] Transcribing {record.file_id} ({record.cell_key})")
        try:
            raw_text = call_for_transcription(client, record, logger)
            if not args.dry_run:
                write_raw_response(record.file_id, raw_text)

            transcript = parse_transcription(record, raw_text, logger)
            if transcript.incomplete_transcript:
                n_incomplete += 1
            n_success += 1

            if args.dry_run:
                print("\n" + "=" * 80)
                print(f"FILE: {record.file_id}  [{record.cell_key}]")
                print("=" * 80)
                print(transcript.model_dump_json(indent=2))
            else:
                write_result(transcript)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            tb = traceback.format_exc()
            logger.error(f"{record.file_id}: FAILED after retries — {exc}\n{tb}")
            if not args.dry_run:
                write_error(record.file_id, str(exc))

    elapsed = time.time() - t0
    logger.info(
        f"Stage 1 done in {elapsed:.1f}s — success={n_success} error={n_error} "
        f"incomplete_transcript={n_incomplete}"
    )
    if args.dry_run:
        logger.info("DRY RUN complete — no files were written. Review output above.")


if __name__ == "__main__":
    main()
