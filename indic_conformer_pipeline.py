"""Indic-Conformer ASR pipeline: chunked transcription + comparison against
the Gemini reference transcripts already in Dostt/**/transcripts/*.json.

UNVERIFIED beyond the pure-Python metrics (asr_metrics.py, unit-tested
separately) — the actual ai4bharat/indic-conformer-600m-multilingual model
call has not been run in this environment. Written against the model's
published usage example; expect to debug real runtime issues yourself.

For each requested chunk size (default 2s/5s/7s/10s, per your test plan):
  1. Load each file's full audio once (torchaudio), resample to 16kHz mono.
  2. Slice into fixed <=chunk_seconds windows (last chunk may be shorter).
  3. Run Indic-Conformer per chunk (CTC by default), giving each chunk a
     timestamp derived from its start offset (chunk_index * chunk_seconds)
     — there is no finer-grained timestamp from the model itself.
  4. Compare the reconstructed transcript against the Gemini reference
     using asr_metrics.py: WER, CER (with S/D/I breakdown), boundary-word
     corruption rate, real-time factor, and policy-critical-term recall.
  5. Write a per-chunk-size metrics CSV plus one final cross-chunk-size
     comparison CSV — the empirical basis for picking a chunk size.

Both the Gemini reference and the Indic-Conformer hypothesis are run
through the SAME asr_metrics.normalize_text() before any comparison, so
formatting differences (punctuation, "[inaudible]" markers, whitespace)
never inflate the measured error rate — see that module's docstring.

Setup (separate venv from the main Gemini pipeline — same one gemma_local.py
uses is fine, it already has torch):
    .venv-gemma/bin/pip install torchaudio
Model download (~600M params) happens automatically on first run.

Usage:
    python3 indic_conformer_pipeline.py --dry-run --dry-run-limit 2
    python3 indic_conformer_pipeline.py --chunk-seconds 2,5,7,10
    python3 indic_conformer_pipeline.py --chunk-seconds 5 --decoding rnnt
    python3 indic_conformer_pipeline.py --force --limit 20
"""
import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import List, Optional

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

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000


def output_dir(chunk_seconds: float) -> Path:
    return config.PROJECT_ROOT / "indic_conformer_results" / f"chunk_{chunk_seconds:g}s"


# ---------------------------------------------------------------------------
# Model loading (lazy — torch/torchaudio only imported when actually run)
# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoModel

    return AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)


def load_audio(path: Path):
    """librosa (not torchaudio.load) — torchaudio's default backend is
    torchcodec, which needs ffmpeg's shared libraries on its exact expected
    rpath; Homebrew's ffmpeg install doesn't sit there and torchcodec can't
    find it. librosa already works reliably for the Gemma pipeline's audio
    handling, so reuse it here instead of fighting torchcodec's linking."""
    import librosa
    import torch

    samples, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)  # librosa resamples internally
    return torch.from_numpy(samples).unsqueeze(0)


def transcribe_file(model, wav, lang_code: str, chunk_seconds: float, decoding: str, logger: StageLogger, file_id: str):
    """Slices the already-loaded, already-resampled waveform into fixed
    windows and transcribes each independently. Returns
    (segments, full_text, per_chunk_times_sec)."""
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    total_samples = wav.shape[-1]

    segments = []
    per_chunk_times = []
    for start_sample in range(0, total_samples, chunk_samples):
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk_wav = wav[:, start_sample:end_sample]
        if chunk_wav.shape[-1] < SAMPLE_RATE * 0.1:  # skip near-empty tail slivers
            continue

        offset_sec = start_sample / SAMPLE_RATE
        t0 = time.time()
        try:
            text = model(chunk_wav, lang_code, decoding)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{file_id} chunk@{offset_sec:.1f}s failed: {exc}\n{traceback.format_exc()}")
            continue
        elapsed = time.time() - t0
        per_chunk_times.append(elapsed)

        if isinstance(text, list):  # some AI4Bharat model wrappers batch-return a list
            text = text[0] if text else ""
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


# ---------------------------------------------------------------------------
# Reference loading (Gemini transcript + classification, already on disk)
# ---------------------------------------------------------------------------
def load_gemini_reference(record: dsv2.FileRecordV2) -> Optional[dict]:
    p = record.transcripts_dir / f"{record.file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    return data


# ---------------------------------------------------------------------------
# Per-file, per-chunk-size orchestration
# ---------------------------------------------------------------------------
def process_file(model, record: dsv2.FileRecordV2, chunk_seconds: float, decoding: str, logger: StageLogger) -> dict:
    lang_code = LANGUAGE_CODES[record.language]
    wav = load_audio(record.path)
    audio_duration_sec = wav.shape[-1] / SAMPLE_RATE

    t0 = time.time()
    segments, full_text, per_chunk_times = transcribe_file(model, wav, lang_code, chunk_seconds, decoding, logger, record.file_id)
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
        "decoding": decoding,
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_per_chunk_report(chunk_seconds: float, results: List[dict]):
    d = output_dir(chunk_seconds)
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        if r["status"] != "success" or r["metrics"] is None:
            continue
        m = r["metrics"]
        rows.append({
            "file_id": r["file_id"],
            "language": r["language"],
            "wer": m["wer"]["wer"],
            "cer": m["cer"]["cer"],
            "boundary_corruption_rate": m["boundary_corruption"]["rate"],
            "policy_term_recall": m["policy_term_recall"]["recall"] if m["policy_term_recall"] else None,
            "real_time_factor": m["real_time_factor"],
        })
    with open(d / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "language", "wer", "cer", "boundary_corruption_rate", "policy_term_recall", "real_time_factor"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_comparison_report(all_rows_by_chunk: dict):
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    comparison = []
    for chunk_seconds, rows in all_rows_by_chunk.items():
        comparison.append({
            "chunk_seconds": chunk_seconds,
            "n_files": len(rows),
            "mean_wer": mean([r["wer"] for r in rows]),
            "mean_cer": mean([r["cer"] for r in rows]),
            "mean_boundary_corruption_rate": mean([r["boundary_corruption_rate"] for r in rows]),
            "mean_policy_term_recall": mean([r["policy_term_recall"] for r in rows]),
            "mean_real_time_factor": mean([r["real_time_factor"] for r in rows]),
        })
    comparison.sort(key=lambda r: r["chunk_seconds"])

    out_path = config.PROJECT_ROOT / "indic_conformer_results" / "chunk_size_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_seconds", "n_files", "mean_wer", "mean_cer", "mean_boundary_corruption_rate", "mean_policy_term_recall", "mean_real_time_factor"])
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-seconds", type=str, default="2,5,7,10")
    parser.add_argument("--decoding", choices=["ctc", "rnnt"], default="ctc")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    chunk_sizes = [float(x) for x in args.chunk_seconds.split(",")]
    logger = StageLogger("indic_conformer")
    logger.info(f"Loading {MODEL_ID}...")
    model = load_model()
    logger.info("Model loaded.")

    all_records = dsv2.load_dataset_v2()
    all_rows_by_chunk = {}

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

        results = []
        n_success, n_error = 0, 0
        t0 = time.time()
        for i, record in enumerate(records, 1):
            logger.info(f"[chunk={chunk_seconds}s {i}/{len(records)}] {record.file_id}")
            try:
                result = process_file(model, record, chunk_seconds, args.decoding, logger)
                n_success += 1
                results.append(result)
                if args.dry_run:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    write_result(chunk_seconds, result)
            except Exception as exc:  # noqa: BLE001
                n_error += 1
                logger.error(f"{record.file_id}: FAILED — {exc}\n{traceback.format_exc()}")
                if not args.dry_run:
                    write_error(chunk_seconds, record.file_id, str(exc))

        logger.info(f"chunk_seconds={chunk_seconds} done in {time.time() - t0:.1f}s — success={n_success} error={n_error}")

        if not args.dry_run:
            rows = write_per_chunk_report(chunk_seconds, results)
            all_rows_by_chunk[chunk_seconds] = rows

    if not args.dry_run and all_rows_by_chunk:
        comparison = write_comparison_report(all_rows_by_chunk)
        print("\n" + "=" * 80)
        print("CHUNK SIZE COMPARISON")
        print("=" * 80)
        for row in comparison:
            print(row)


if __name__ == "__main__":
    main()
