"""Full-dataset confidence analysis across every scored run, per model and
flag category, on the model's self-reported confidence (model_confidence):

  1. confidence_buckets_by_category.csv — FLAG level, per confidence bucket
     [0.0,0.1), [0.1,0.2), ..., [0.9,1.0] (+ "missing" for flags with no
     confidence): TP / redundant / FP flag counts and precision. Same flag
     rules as threshold_analysis.py's flag_tp/flag_fp: a flag belongs to the
     category it labelled itself with; TP = matched to a GT flag; redundant =
     unmatched but a verified extra instance of an already-matched violation
     (counted as not-an-error in precision, as everywhere else); FP = the rest.
     So bucket TP+redundant+FP summed over buckets = that model's flag totals.

  2. threshold_08_09_by_category.csv — FILE level at confidence >= 0.8 and
     >= 0.9: TP/TN/FP/FN, precision, recall, specificity, accuracy (+F1),
     computed by threshold_analysis.threshold_metrics_for_group itself, so the
     numbers are identical to the existing threshold_metrics_by_category rows.

Read-only over analyze_results.py's per_file caches — no scoring, no API calls.
Partial runs are included on the files they finished (n_files column).

Usage: python3 confidence_bucket_analysis.py [--out-dir analysis_results_full_summary]
"""
import argparse
import csv
from pathlib import Path

import config
from threshold_analysis import _is_redundant, load_cached_rows, threshold_metrics_for_group

# (prompt label, analysis dir) for every full-dataset analysis; every model
# with a per_file/ cache inside is included.
RUNS = [
    ("v1", "analysis_results"),
    ("v6", "analysis_results_promptv6"),
    ("v7", "analysis_results_promptv7"),
    ("v8", "analysis_results_promptv8"),
]
CATEGORIES = ["PlatformMove", "SuspiciousActivity", "Explicit-Flirting", "ALL"]
BUCKETS = [(i / 10, (i + 1) / 10) for i in range(10)]
THRESHOLDS = [0.8, 0.9]
TOTAL_FILES = 381


def bucket_of(c):
    if c is None:
        return "missing"
    for lo, hi in BUCKETS:
        if lo <= c < hi or (hi == 1.0 and c == 1.0):
            return f"{lo:.1f}-{hi:.1f}"
    return "out_of_range"


def r2(x):
    return round(x, 2) if x is not None else None


def bucket_rows(prompt, model, rows):
    out = []
    for cat in CATEGORIES:
        counts = {}
        for r in rows:
            for f in r["model_flags"]:
                if cat != "ALL" and f.get("category") != cat:
                    continue
                b = counts.setdefault(bucket_of(f.get("model_confidence")), {"tp": 0, "redundant": 0, "fp": 0})
                if f.get("matched"):
                    b["tp"] += 1
                elif _is_redundant(f):
                    b["redundant"] += 1
                else:
                    b["fp"] += 1
        labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in BUCKETS] + ["missing", "out_of_range"]
        for label in labels:
            if label in ("missing", "out_of_range") and label not in counts:
                continue
            c = counts.get(label, {"tp": 0, "redundant": 0, "fp": 0})
            n = c["tp"] + c["redundant"] + c["fp"]
            out.append({
                "prompt": prompt, "model": model, "n_files": len(rows), "category": cat, "confidence_bucket": label,
                "n_flags": n, "tp": c["tp"], "redundant": c["redundant"], "fp": c["fp"],
                "precision": r2((c["tp"] + c["redundant"]) / n) if n else None,
            })
    return out


def threshold_rows(prompt, model, rows):
    out = []
    for cat in CATEGORIES:
        for t in THRESHOLDS:
            m = threshold_metrics_for_group(rows, cat, t, "model_confidence")
            out.append({
                "prompt": prompt, "model": model, "n_files": len(rows), "category": cat, "threshold": f">={t}",
                **{k: m[k] for k in ("n_gt_pos", "n_gt_neg", "tp", "fp", "fn", "tn",
                                      "precision", "recall", "specificity", "accuracy", "f1")},
            })
    return out


def write(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis_results_full_summary")
    out_dir = config.PROJECT_ROOT / parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets, thresholds = [], []
    for prompt, adir in RUNS:
        base = config.PROJECT_ROOT / adir
        for model_dir in sorted(p.parent for p in base.glob("*/per_file") if p.is_dir()):
            rows = load_cached_rows(base, model_dir.name)
            if not rows:
                continue
            buckets += bucket_rows(prompt, model_dir.name, rows)
            thresholds += threshold_rows(prompt, model_dir.name, rows)
            note = "" if len(rows) >= TOTAL_FILES else f"  (partial: {len(rows)}/{TOTAL_FILES} files)"
            print(f"{prompt} {model_dir.name}: {len(rows)} files{note}")

    write(out_dir / "confidence_buckets_by_category.csv", buckets)
    write(out_dir / "threshold_08_09_by_category.csv", thresholds)
    print(f"Wrote {out_dir}/confidence_buckets_by_category.csv and threshold_08_09_by_category.csv")


if __name__ == "__main__":
    main()
