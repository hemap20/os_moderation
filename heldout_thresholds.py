"""Held-out thresholds and scoring modes: for every full-dataset run, on the
HELD-OUT files only (full minus manifest/dev_set_v1.csv), file-level
recall/specificity (category ALL) for each scoring mode x threshold, overall
and per audio bucket (audio_segments.csv: length short/long, quality
good/bad, difficulty clear/ambiguous violation).

Read-only; reuses heldout_leaderboard's helpers and threshold_analysis's
metric rules, so numbers match the held-out leaderboard.

Usage: python3 heldout_thresholds.py
Writes analysis_results_full_summary/heldout_thresholds.csv
"""
import csv

import config
from heldout_leaderboard import DEV_IDS, N_HELDOUT, metrics
from leaderboard import full_dataset_runs, load_rows

MODES = ["raw", "sa", "sa_drop_invalid", "pp_full"]
THRESHOLDS = [None, 0.3, 0.6, 0.8, 0.9]
SEGMENTS = {r["file_id"]: r for r in csv.DictReader(open(config.PROJECT_ROOT / "analysis_results_full_summary" / "audio_segments.csv"))}
BUCKETS = [("all", None, None),
           ("short", "length_segment", "short"), ("long", "length_segment", "long"),
           ("good", "quality_segment", "good"), ("bad", "quality_segment", "bad")]


def main():
    out = []
    for prompt, folder in full_dataset_runs():
        for model_dir in sorted(p.parent for p in folder.glob("*/per_file") if p.is_dir()):
            model = model_dir.name
            rows = [r for r in load_rows(folder, model) if r["file_id"] not in DEV_IDS]
            if not rows:
                continue
            modes = ["raw"] if prompt == "v1" else MODES
            for bucket, axis, value in BUCKETS:
                brows = rows if axis is None else [r for r in rows if SEGMENTS.get(r["file_id"], {}).get(axis) == value]
                for mode in modes:
                    for t in THRESHOLDS:
                        rec, spec = metrics(brows, mode, t, prompt)
                        out.append({"model": model, "prompt": prompt, "heldout_files": len(rows),
                                    "complete": len(rows) >= N_HELDOUT - 2, "bucket": bucket,
                                    "bucket_files": len(brows), "mode": mode,
                                    "threshold": "none" if t is None else t, "recall": rec, "specificity": spec})
            # Difficulty: recall only (clean files have no difficulty).
            for value in ("clear_violation", "ambiguous_violation"):
                brows = [r for r in rows if SEGMENTS.get(r["file_id"], {}).get("difficulty_segment") == value]
                for mode in modes:
                    for t in THRESHOLDS:
                        rec, _ = metrics(brows, mode, t, prompt)
                        out.append({"model": model, "prompt": prompt, "heldout_files": len(rows),
                                    "complete": len(rows) >= N_HELDOUT - 2, "bucket": value,
                                    "bucket_files": len(brows), "mode": mode,
                                    "threshold": "none" if t is None else t, "recall": rec, "specificity": None})
    path = config.PROJECT_ROOT / "analysis_results_full_summary" / "heldout_thresholds.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
