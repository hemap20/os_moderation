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
     n_gt_flags = ground-truth flags of that category in the files the model
     was scored on (same value on every bucket row; GT has no confidence).
     n_gt_pos_files / n_gt_neg_files = files with / without a GT flag of that
     category — the GT-negative side, since a flag-level TN doesn't exist.

  2. threshold_08_09_by_category.csv — FILE level at confidence >= 0.8 and
     >= 0.9: TP/TN/FP/FN, precision, recall, specificity, accuracy (+F1),
     computed by threshold_analysis.threshold_metrics_for_group itself, so the
     numbers are identical to the existing threshold_metrics_by_category rows.

Read-only over analyze_results.py's per_file caches — no scoring, no API calls.
Partial runs are included on the files they finished (n_files column).

--scoring-mode pp_full replaces each flag's model_confidence with its
scoring_modes pp_full value (flags scored None are dropped), via
threshold_by_bucket_full.ensure_scored / rows_for_mode, and writes the same
three CSVs with a _pp_full suffix. v1 runs have no pp_full and are skipped.

Usage: python3 confidence_bucket_analysis.py [--out-dir analysis_results_full_summary] [--scoring-mode raw|pp_full]
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
        # GT flags of this category across the files this model was scored on
        # — per model, since partial runs cover fewer files. Not split by
        # bucket: a GT flag has no model confidence of its own.
        n_gt_flags = sum(len([g for g in r.get("gt_flags", []) if cat == "ALL" or g.get("category") == cat]) for r in rows)
        # File counts: TN has no flag-level meaning, so the GT-negative side is
        # given as files with no GT flag of this category (same as n_gt_neg in
        # the threshold tables).
        n_gt_pos_files = sum(1 for r in rows if any(cat == "ALL" or g.get("category") == cat for g in r.get("gt_flags", [])))
        n_gt_neg_files = len(rows) - n_gt_pos_files
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
                "prompt": prompt, "model": model, "n_files": len(rows), "category": cat, "n_gt_flags": n_gt_flags,
                "n_gt_pos_files": n_gt_pos_files, "n_gt_neg_files": n_gt_neg_files, "confidence_bucket": label,
                "n_flags": n, "tp": c["tp"], "redundant": c["redundant"], "fp": c["fp"],
                "precision": r2((c["tp"] + c["redundant"]) / n) if n else None,
            })
    return out


def _max_conf(flags):
    """Highest model_confidence among `flags`, or "missing" if they all lack one."""
    vals = [f.get("model_confidence") for f in flags if f.get("model_confidence") is not None]
    return max(vals) if vals else None


def file_bucket_rows(prompt, model, rows):
    """FILE-level version of bucket_rows, consistent with
    threshold_metrics_for_group: a GT-positive file is a TP in the bucket of
    its highest-confidence flag that matched a GT flag of this category (for
    ALL: any matched flag); a GT-negative file is an FP in the bucket of its
    highest-confidence flag self-labelled as this category. Summing buckets
    from the top down to threshold t gives exactly the threshold table's
    TP/FP at t. GT-positive files with no such matched flag are FN at every
    threshold, and GT-negative files with no such flag are TN at every
    threshold — neither has a bucket, so both appear only in the totals."""
    out = []
    labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in BUCKETS] + ["missing"]
    for cat in CATEGORIES:
        tp, fp = {}, {}
        n_pos = n_neg = never_caught = never_flagged = 0
        for r in rows:
            gt = [g for g in r.get("gt_flags", []) if cat == "ALL" or g.get("category") == cat]
            flags = r["model_flags"]
            if gt:
                n_pos += 1
                caught = [f for f in flags if f.get("matched") and (cat == "ALL" or f.get("matched_gt_category") == cat)]
                if caught:
                    b = bucket_of(_max_conf(caught))
                    tp[b] = tp.get(b, 0) + 1
                else:
                    never_caught += 1
            else:
                n_neg += 1
                own = [f for f in flags if cat == "ALL" or f.get("category") == cat]
                if own:
                    b = bucket_of(_max_conf(own))
                    fp[b] = fp.get(b, 0) + 1
                else:
                    never_flagged += 1
        for label in labels:
            t, f_ = tp.get(label, 0), fp.get(label, 0)
            if label == "missing" and not (t or f_):
                continue
            out.append({
                "prompt": prompt, "model": model, "n_files": len(rows), "category": cat,
                "n_gt_pos_files": n_pos, "n_gt_neg_files": n_neg,
                "fn_files_never_caught": never_caught, "tn_files_never_flagged": never_flagged,
                "confidence_bucket": label, "tp_files": t, "fp_files": f_,
                "precision": r2(t / (t + f_)) if (t + f_) else None,
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


def apply_scoring_mode(rows, prompt, mode):
    """rows with model_confidence replaced by the `mode` signal; flags the
    mode drops are removed. raw returns rows unchanged."""
    if mode == "raw":
        return rows
    from threshold_by_bucket_full import ensure_scored, rows_for_mode
    ensure_scored(rows, prompt, [mode])
    out = rows_for_mode(rows, mode)
    for r in out:
        r["model_flags"] = [{**f, "model_confidence": f["_scored_signal"]} for f in r["model_flags"]]
    return out


def write(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis_results_full_summary")
    parser.add_argument("--scoring-mode", choices=["raw", "pp_full"], default="raw")
    args = parser.parse_args()
    out_dir = config.PROJECT_ROOT / args.out_dir
    mode = args.scoring_mode
    suffix = "" if mode == "raw" else f"_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets, file_buckets, thresholds = [], [], []
    for prompt, adir in RUNS:
        if mode != "raw" and prompt == "v1":
            continue
        base = config.PROJECT_ROOT / adir
        for model_dir in sorted(p.parent for p in base.glob("*/per_file") if p.is_dir()):
            rows = load_cached_rows(base, model_dir.name)
            if not rows:
                continue
            rows = apply_scoring_mode(rows, prompt, mode)
            buckets += bucket_rows(prompt, model_dir.name, rows)
            file_buckets += file_bucket_rows(prompt, model_dir.name, rows)
            thresholds += threshold_rows(prompt, model_dir.name, rows)
            note = "" if len(rows) >= TOTAL_FILES else f"  (partial: {len(rows)}/{TOTAL_FILES} files)"
            print(f"{prompt} {model_dir.name}: {len(rows)} files{note}")

    write(out_dir / f"confidence_buckets_by_category{suffix}.csv", buckets)
    write(out_dir / f"threshold_08_09_by_category{suffix}.csv", thresholds)
    write(out_dir / f"confidence_buckets_by_category_file_level{suffix}.csv", file_buckets)
    print(f"Wrote {out_dir}/confidence_buckets_by_category{suffix}.csv, confidence_buckets_by_category_file_level{suffix}.csv "
          f"and threshold_08_09_by_category{suffix}.csv")


if __name__ == "__main__":
    main()
