"""Builds threshold_pivot_<model>.csv from threshold_analysis.py's
threshold_metrics_by_category.csv: one row per (category, threshold), with
the file-level confusion block for each confidence signal side by side
(Model Confidence Score | Log Probability | Token Entropy). Read-only over
that CSV — never re-scores anything.

Usage:
    python3 threshold_pivot.py analysis_results_dev_promptv8 [more dirs...]
"""
import argparse
import csv
from pathlib import Path

SIGNALS = [
    ("model_confidence", "Model Confidence Score"),
    ("logprob_derived_confidence", "Log Probability"),
    ("neg_entropy", "Token Entropy"),
]
CATEGORIES = ["Explicit-Flirting", "PlatformMove", "SuspiciousActivity", "ALL"]
THRESHOLDS = [f"{t / 10:.1f}" for t in range(10)] + ["none"]
# neg_entropy uses its own grid (threshold_analysis.ENTROPY_THRESHOLDS), so its
# block is aligned by row POSITION. This mapping reproduces the existing pivot
# files exactly (checked against all 25 of them): rows 0.1..0.9 show entropy
# thresholds -2.0..0.0, and row 0.0 ALSO shows 0.0 — so -4.0 never appears and
# rows 0.0 and 0.9 carry the same entropy numbers.
ENTROPY_ROW_THRESHOLDS = ["0.0", "-2.0", "-1.0", "-0.5", "-0.25", "-0.1", "-0.05", "-0.02", "-0.01", "0.0", "none"]
BLOCK = ["tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"]


def write_pivots(analysis_dir: Path) -> list:
    src = analysis_dir / "threshold_metrics_by_category.csv"
    rows = [r for r in csv.DictReader(src.open()) if r["language"] == "ALL"]
    by_key = {(r["model"], r["signal"], r["category"], r["threshold"]): r for r in rows}
    written = []
    for model in sorted({r["model"] for r in rows}):
        out = analysis_dir / f"threshold_pivot_{model}.csv"
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            header_top = ["", "", "", ""]
            for _, label in SIGNALS:
                header_top += [label] + [""] * (len(BLOCK) - 1)
            w.writerow(header_top)
            w.writerow(["category", "threshold", "n_gt_pos", "n_gt_neg"] + BLOCK * len(SIGNALS))
            for cat in CATEGORIES:
                for th in THRESHOLDS:
                    base = by_key.get((model, SIGNALS[0][0], cat, th))
                    if base is None:
                        continue
                    line = [cat, th, base["n_gt_pos"], base["n_gt_neg"]]
                    for sig, _ in SIGNALS:
                        sig_th = ENTROPY_ROW_THRESHOLDS[THRESHOLDS.index(th)] if sig == "neg_entropy" else th
                        r = by_key.get((model, sig, cat, sig_th), {})
                        line += [r.get(k, "") for k in BLOCK]
                    w.writerow(line)
        written.append(out)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dirs", nargs="+")
    for d in parser.parse_args().analysis_dirs:
        for out in write_pivots(Path(d)):
            print(out)


if __name__ == "__main__":
    main()
