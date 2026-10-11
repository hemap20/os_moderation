"""Per-file noise-level estimate for the audio-quality bucket, replacing
the earlier WER-based proxy with a direct signal measurement.

Method: decode each file to 16kHz mono PCM via ffmpeg, split into 32ms
frames, take each frame's RMS level in dB. The 10th percentile of frame
RMS approximates the background noise floor (the quietest moments — gaps
between words/sentences — are dominated by whatever background noise is
present); the 90th percentile approximates the speech peak level. Their
difference is an SNR proxy: a clean recording has quiet gaps and loud
speech (high SNR); a noisy recording's "quiet" moments are still loud
with background noise (low SNR).

Writes analysis_results_full_summary/audio_snr.csv:
  file_id, noise_floor_db, speech_level_db, snr_db
"""
import glob
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import analyze_results as ar

OUT_DIR = Path("analysis_results_full_summary")
OUT_DIR.mkdir(exist_ok=True)


def find_audio_path(file_id: str):
    matches = glob.glob(f"Dostt/**/{file_id}", recursive=True)
    return matches[0] if matches else None


def frame_snr_db(path: str, frame_ms: int = 32, sr: int = 16000):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames < 10:
        return None
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    noise_floor_db = float(np.percentile(rms_db, 10))
    speech_db = float(np.percentile(rms_db, 90))
    return noise_floor_db, speech_db, speech_db - noise_floor_db


def process_one(file_id):
    path = find_audio_path(file_id)
    if path is None:
        return {"file_id": file_id, "noise_floor_db": "", "speech_level_db": "", "snr_db": ""}
    result = frame_snr_db(path)
    if result is None:
        return {"file_id": file_id, "noise_floor_db": "", "speech_level_db": "", "snr_db": ""}
    noise_floor_db, speech_db, snr_db = result
    return {"file_id": file_id, "noise_floor_db": round(noise_floor_db, 2),
            "speech_level_db": round(speech_db, 2), "snr_db": round(snr_db, 2)}


def main():
    file_ids = []
    for fp in glob.glob("analysis_results/e2b_nothinking/per_file/*.json"):
        d = json.loads(Path(fp).read_text())
        file_ids.append(d["file_id"])
    file_ids = sorted(set(file_ids))
    print(f"Computing SNR for {len(file_ids)} files...")

    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_one, fid): fid for fid in file_ids}
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(file_ids)}")

    rows.sort(key=lambda r: r["file_id"])
    ar.write_csv(OUT_DIR / "audio_snr.csv", rows, ["file_id", "noise_floor_db", "speech_level_db", "snr_db"])
    n_missing = sum(1 for r in rows if r["snr_db"] == "")
    print(f"wrote audio_snr.csv ({len(rows)} rows, {n_missing} missing)")


if __name__ == "__main__":
    main()
