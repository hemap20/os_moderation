"""Indic-Transcribe-Core ASR pipeline — same chunked-transcription +
Gemini-reference-comparison structure as indic_conformer_pipeline.py, using
bodhan-ai/indic-transcribe-core (1B params, NVIDIA Canary-2 FastConformer
AED architecture) instead of ai4bharat/indic-conformer-600m-multilingual.

UNVERIFIED beyond the pure-Python metrics (asr_metrics.py, already
unit-tested) — the actual model call has not been run in this environment.
Written against the model card's documented usage; expect to debug real
runtime issues yourself.

Key differences from indic_conformer_pipeline.py:
  - Loaded via a custom IndicTranscribe class (huggingface_hub.snapshot_download
    + sys.path insert), not AutoModel/AutoProcessor.
  - 45s/call hard limit (vs no documented limit for Indic-Conformer) — your
    chunk sizes (2/5/7/10s) are all well under this, so it's not a practical
    constraint here, just noted for completeness.
  - Native-script-only output (same as what we need).
  - Call signature: asr(audio_path_or_array, lang="hi") -> str (plain text,
    no segments) — same as Indic-Conformer's raw call, so the same
    fixed-window chunking + chunk-offset timestamp derivation applies.

Usage:
    python3 indic_transcribe_core_pipeline.py --dry-run --dry-run-limit 2
    python3 indic_transcribe_core_pipeline.py --chunk-seconds 2,5,7,10
    python3 indic_transcribe_core_pipeline.py --force --limit 20
"""
import argparse
import csv
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import asr_metrics
import config
import dataset_v2 as dsv2
from pipeline_logging import StageLogger

LANGUAGE_CODES = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
}

MODEL_ID = "bodhan-ai/indic-transcribe-core"
SAMPLE_RATE = 16000
MODEL_CALL_LIMIT_SEC = 45.0  # hard limit per the model card; our chunk sizes are all well under this


def output_dir(chunk_seconds: float) -> Path:
    return config.PROJECT_ROOT / "indic_transcribe_core_results" / f"chunk_{chunk_seconds:g}s"


# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------
def load_model():
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(MODEL_ID)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from indic_transcribe import IndicTranscribe  # noqa: E402 — only importable after the path insert above

    return IndicTranscribe.from_pretrained(model_dir)


def load_audio(path: Path):
    import librosa

    samples, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return samples


def transcribe_file(asr, wav, lang_code: str, chunk_seconds: float, logger: StageLogger, file_id: str):
    if chunk_seconds > MODEL_CALL_LIMIT_SEC:
        raise ValueError(f"chunk_seconds={chunk_seconds} exceeds the model's {MODEL_CALL_LIMIT_SEC}s/call limit")

    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    total_samples = len(wav)

    segments = []
    per_chunk_times = []
    for start_sample in range(0, total_samples, chunk_samples):
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk_wav = wav[start_sample:end_sample]
        if len(chunk_wav) < SAMPLE_RATE * 0.1:
            continue

        offset_sec = start_sample / SAMPLE_RATE
        t0 = time.time()
        try:
            text = asr(chunk_wav, lang=lang_code)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{file_id} chunk@{offset_sec:.1f}s failed: {exc}\n{traceback.format_exc()}")
            continue
        elapsed = time.time() - t0
        per_chunk_times.append(elapsed)

        if isinstance(text, tuple):  # in case return_lid-style calls sneak in a tuple
            text = text[0]
        segments.append({"t": _format_mmss(offset_sec), "text": str(text).strip()})

    full_text = " ".join(s["text"] for s in segments)
    return segments, full_text, per_chunk_times


def _format_mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _parse_mmss(ts: str) -> Optional[float]:
    try:
        parts = [float(p) for p in ts.strip().split(":")]
    except Exception:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def load_gemini_reference(record: dsv2.FileRecordV2) -> Optional[dict]:
    p = record.transcripts_dir / f"{record.file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    return data


def process_file(asr, record: dsv2.FileRecordV2, chunk_seconds: float, logger: StageLogger) -> dict:
    lang_code = LANGUAGE_CODES[record.language]
    wav = load_audio(record.path)
    audio_duration_sec = len(wav) / SAMPLE_RATE

    t0 = time.time()
    segments, full_text, per_chunk_times = transcribe_file(asr, wav, lang_code, chunk_seconds, logger, record.file_id)
    total_time = time.time() - t0

    reference = load_gemini_reference(record)
    metrics = None
    if reference is not None:
        ref_full_text = reference["transcript"].get("full_text", "")
        ref_segment_starts = [
            t for t in (_parse_mmss(s.get("t", "")) for s in reference["transcript"].get("segments", []))
            if t is not None
        ]
        excerpts = [f.get("ground_truth_excerpt", "") for f in reference.get("classification", {}).get("ground_truth_flags", [])]

        metrics = {
            "wer": asr_metrics.word_error_rate(ref_full_text, full_text),
            "cer": asr_metrics.char_error_rate(ref_full_text, full_text),
            "boundary_corruption": asr_metrics.boundary_corruption_rate(chunk_seconds, audio_duration_sec, ref_segment_starts),
            "policy_term_recall": asr_metrics.policy_term_recall(excerpts, full_text) if excerpts else None,
            "real_time_factor": total_time / audio_duration_sec if audio_duration_sec else None,
        }
    else:
        logger.warn(f"{record.file_id}: no Gemini reference available, skipping metrics")

    return {
        "file_id": record.file_id,
        "language": record.language,
        "chunk_seconds": chunk_seconds,
        "audio_duration_sec": audio_duration_sec,
        "processing_time_sec": total_time,
        "segments": segments,
        "full_text": full_text,
        "metrics": metrics,
        "status": "success",
    }


def already_done(chunk_seconds: float, file_id: str) -> bool:
    p = output_dir(chunk_seconds) / "transcripts" / f"{file_id}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "success"
    except Exception:
        return False


def write_result(chunk_seconds: float, result: dict):
    d = output_dir(chunk_seconds) / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result['file_id']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


def write_error(chunk_seconds: float, file_id: str, error: str):
    d = output_dir(chunk_seconds) / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{file_id}.json").write_text(json.dumps({"file_id": file_id, "status": "error", "error": error}, indent=2))


def write_per_chunk_report(chunk_seconds: float):
    """Reads ALL completed transcripts for this chunk size from disk — not
    just whatever subset this particular invocation processed — so resuming
    a partially- or fully-done chunk size still produces a report covering
    every file, not only the newly-processed ones."""
    d = output_dir(chunk_seconds)
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in (d / "transcripts").glob("*.json"):
        r = json.loads(p.read_text())
        if r.get("status") != "success" or r.get("metrics") is None:
            continue
        m = r["metrics"]
        rows.append({
            "file_id": r["file_id"], "language": r["language"],
            "wer": m["wer"]["wer"], "cer": m["cer"]["cer"],
            "wer_ref_word_count": m["wer"]["ref_word_count"],
            "cer_ref_char_count": m["cer"]["ref_char_count"],
            "boundary_corruption_rate": m["boundary_corruption"]["rate"],
            "policy_term_recall": m["policy_term_recall"]["recall"] if m["policy_term_recall"] else None,
            "real_time_factor": m["real_time_factor"],
        })
    with open(d / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "language", "wer", "cer", "wer_ref_word_count", "cer_ref_char_count", "boundary_corruption_rate", "policy_term_recall", "real_time_factor"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_comparison_report(chunk_sizes):
    """WER/CER are MICRO-averaged (total errors / total reference
    words-or-chars across the corpus), not a mean of per-file ratios — a
    simple mean gets badly distorted by near-silent calls where the
    reference is only 1-3 words, since a single genuine filler word there
    can register as WER>10 for that one file alone. Other metrics (already
    bounded ratios, not error-count-over-tiny-denominator) still use a
    plain mean."""
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def micro_average(rows, err_key, ref_key):
        total_err, total_ref = 0.0, 0
        for r in rows:
            if r[ref_key]:
                total_err += r[err_key] * r[ref_key]  # rows store the ratio; recover the error count
                total_ref += r[ref_key]
        return total_err / total_ref if total_ref else None

    comparison = []
    for chunk_seconds in chunk_sizes:
        rows = write_per_chunk_report(chunk_seconds)
        comparison.append({
            "chunk_seconds": chunk_seconds, "n_files": len(rows),
            "micro_wer": micro_average(rows, "wer", "wer_ref_word_count"),
            "micro_cer": micro_average(rows, "cer", "cer_ref_char_count"),
            "mean_boundary_corruption_rate": mean([r["boundary_corruption_rate"] for r in rows]),
            "mean_policy_term_recall": mean([r["policy_term_recall"] for r in rows]),
            "mean_real_time_factor": mean([r["real_time_factor"] for r in rows]),
        })
    comparison.sort(key=lambda r: r["chunk_seconds"])

    out_path = config.PROJECT_ROOT / "indic_transcribe_core_results" / "chunk_size_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_seconds", "n_files", "micro_wer", "micro_cer", "mean_boundary_corruption_rate", "mean_policy_term_recall", "mean_real_time_factor"])
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


def unique_records():
    seen = set()
    for r in dsv2.load_dataset_v2():
        if r.file_id in seen:
            continue
        seen.add(r.file_id)
        yield r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-seconds", type=str, default="2,5,7,10")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4,
                         help="Concurrent files in flight (GPU-bound, but low GPU utilization from one-at-a-time "
                              "dispatch overhead means threading can still help keep the GPU busier — PyTorch "
                              "releases the GIL during actual kernel execution)")
    args = parser.parse_args()

    chunk_sizes = [float(x) for x in args.chunk_seconds.split(",")]
    logger = StageLogger("indic_transcribe_core")
    logger.info(f"Loading {MODEL_ID}...")
    asr = load_model()
    logger.info("Model loaded.")

    all_records = list(unique_records())
    processed_chunk_sizes = []

    for chunk_seconds in chunk_sizes:
        records = all_records
        if args.dry_run:
            records = records[: args.dry_run_limit]
            logger.info(f"DRY RUN chunk_seconds={chunk_seconds}: {len(records)} file(s)")
        else:
            if not args.force:
                before = len(records)
                records = [r for r in records if not already_done(chunk_seconds, r.file_id)]
                logger.info(f"chunk_seconds={chunk_seconds}: skipping {before - len(records)} already-done file(s)")
            if args.limit:
                records = records[: args.limit]

        n_success, n_error = 0, 0
        counts_lock = threading.Lock()
        t0 = time.time()

        def process_one(i, record):
            nonlocal n_success, n_error
            logger.info(f"[chunk={chunk_seconds}s {i}/{len(records)}] {record.file_id}")
            try:
                result = process_file(asr, record, chunk_seconds, logger)
                with counts_lock:
                    n_success += 1
                if args.dry_run:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    write_result(chunk_seconds, result)
            except Exception as exc:  # noqa: BLE001
                with counts_lock:
                    n_error += 1
                logger.error(f"{record.file_id}: FAILED — {exc}\n{traceback.format_exc()}")
                if not args.dry_run:
                    write_error(chunk_seconds, record.file_id, str(exc))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_one, i, record) for i, record in enumerate(records, 1)]
            for fut in as_completed(futures):
                fut.result()  # re-raise anything that escaped process_one's own try/except

        logger.info(f"chunk_seconds={chunk_seconds} done in {time.time() - t0:.1f}s — success={n_success} error={n_error}")

        if not args.dry_run:
            processed_chunk_sizes.append(chunk_seconds)

    if not args.dry_run and processed_chunk_sizes:
        comparison = write_comparison_report(processed_chunk_sizes)
        print("\n" + "=" * 80)
        print("CHUNK SIZE COMPARISON")
        print("=" * 80)
        for row in comparison:
            print(row)


if __name__ == "__main__":
    main()
