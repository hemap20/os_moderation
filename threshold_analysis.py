"""Threshold-by-category metrics — a SEPARATE analysis pass from
analyze_results.py, deliberately kept out of it so nothing here can ever
change analyze_results.py's own tables (confusion_file_level_overall.csv,
confusion_flag_level_overall.csv, etc.) or its scoring behavior.

Read-only over analyze_results.py's ALREADY-SCORED per-file cache
(analysis_results*/{model}/per_file/*.json) — this script never re-runs
matching, never calls Gemini, and never touches gemma_results/gemini_results.
It only adds ONE new file per output directory: threshold_metrics_by_category.csv.

Definitions (exact):
  - Thresholds: --thresholds, default 0.0..0.9 step 0.1, plus an always-added
    "none" row (every flag counts, including missing-confidence ones).
  - A flag with missing model_confidence counts as below every NUMERIC
    threshold (never passes), but counts in the "none" row.
  - Categories: PlatformMove, SuspiciousActivity, Explicit-Flirting, ALL.
  - File-level TP/FN (the main, STRICT rule): gt_pos(C) AND at least one
    model flag at/above the threshold matched==true to a GT flag of category
    C (via matched_gt_category, recorded by analyze_results.py's score_file
    from the real match pairing). For ALL, no category condition on the
    match — exactly file_bucket's rule, so the "none"+ALL row is required to
    reproduce confusion_file_level_overall/by_language and confusion_flag_level
    exactly (checked below).
  - File-level FP/TN: gt_neg(C) AND at least one model flag of category C
    (ALL: any flag) at/above the threshold — NOT conditioned on matching.
  - loose_tp/loose_recall: the coarse gt_pos(C) AND "any model flag of
    category C at/above threshold" reading (no match required) — shows how
    often the model is right about the file but for the wrong reason.
  - Flag-level: flag_tp/flag_fp use the model's OWN category label (not the
    matched GT's), at/above threshold; flag_fn = GT flags of category C -
    flag_tp. No flag-level TN/specificity/accuracy (undefined at flag level).

Usage:
    python3 threshold_analysis.py --dataset-root Dostt_dev --results-tag promptv5
    python3 threshold_analysis.py --thresholds 0.0,0.25,0.5,0.75
"""
import argparse
import glob
import json
from pathlib import Path
from typing import List, Optional

import config
from analyze_results import (
    aggregate_file_level,
    aggregate_flag_level,
    configure_dataset_root,
    merge_and_write_csv,
    r2,
)
import analyze_results as ar
from pipeline_logging import StageLogger

DEFAULT_THRESHOLDS = [round(i * 0.1, 1) for i in range(10)]  # 0.0, 0.1, ..., 0.9
THRESHOLD_TABLE_CATEGORIES = list(config.CATEGORY_LABELS.values()) + ["ALL"]


def _passes_threshold(flag: dict, threshold: Optional[float]) -> bool:
    """threshold=None means the 'none' row: every flag counts, including
    missing-confidence ones (matches confusion_file_level_overall's raw
    behavior). Otherwise a missing model_confidence counts as below EVERY
    threshold, per spec — never treated as passing."""
    if threshold is None:
        return True
    c = flag.get("model_confidence")
    if c is None:
        return False
    return c >= threshold


def threshold_metrics_for_group(rows: List[dict], category: str, threshold: Optional[float]) -> dict:
    """One row of threshold_metrics_by_category.csv, for one (language,
    category, threshold) triple — `rows` is already scoped to one language
    (or all languages, for the "ALL" language row) by the caller.

    TP/FN (the main rule) are STRICT: a model flag only counts toward TP if
    it actually matched a ground-truth flag OF THE SAME CATEGORY (via each
    model flag's "matched_gt_category", set in analyze_results.py's
    score_file from the real match pairing) — a PlatformMove flag matched
    to an Explicit-Flirting GT flag is not a PlatformMove TP. For ALL
    there's no category condition on the match itself, so ALL's TP/FN is
    EXACTLY file_bucket's rule. FP/TN stay on the coarse "any flag of this
    category passed the threshold" rule (matching was never a precision
    concern — a spurious flag in a GT-negative file is still a false
    positive regardless of what it "matched"). loose_tp/loose_recall keep
    the old category+confidence-only reading for comparison (see how often
    the model is "right about the file but for the wrong reason")."""
    tp = fp = fn = tn = 0
    loose_tp = 0
    n_gt_pos = 0
    n_gt_neg = 0
    flag_tp = 0
    flag_fp = 0
    n_flags_missing_confidence = 0
    total_gt_of_cat = 0

    for r in rows:
        gt_flags = r.get("gt_flags", [])
        model_flags = r["model_flags"]

        if category == "ALL":
            gt_of_cat = gt_flags
            model_of_cat_any_conf = model_flags
        else:
            gt_of_cat = [g for g in gt_flags if g.get("category") == category]
            model_of_cat_any_conf = [f for f in model_flags if f.get("category") == category]

        total_gt_of_cat += len(gt_of_cat)
        n_flags_missing_confidence += sum(1 for f in model_of_cat_any_conf if f.get("model_confidence") is None)

        model_of_cat = [f for f in model_of_cat_any_conf if _passes_threshold(f, threshold)]
        # Strict population is threshold-gated but NOT category-gated on the
        # model's own label — for a named category, a model flag of ANY
        # category that matched a GT flag OF THIS category still counts
        # (the model's own mislabeled category shouldn't hide a real catch);
        # for ALL, any matched flag counts, exactly file_bucket's rule.
        strict_pool = [f for f in model_flags if _passes_threshold(f, threshold)]

        gt_pos = len(gt_of_cat) > 0
        loose_pos = len(model_of_cat) > 0
        if category == "ALL":
            strict_pos = any(f.get("matched") for f in strict_pool)
        else:
            strict_pos = any(f.get("matched") and f.get("matched_gt_category") == category for f in strict_pool)
        fp_pos = loose_pos  # FP/TN use the coarse "any flag of this category" reading, unaffected by strict/loose

        if gt_pos:
            n_gt_pos += 1
        else:
            n_gt_neg += 1

        if gt_pos and strict_pos:
            tp += 1
        elif gt_pos and not strict_pos:
            fn += 1
        elif not gt_pos and fp_pos:
            fp += 1
        else:
            tn += 1

        if gt_pos and loose_pos:
            loose_tp += 1

        flag_tp += sum(1 for f in model_of_cat if f.get("matched"))
        flag_fp += sum(1 for f in model_of_cat if not f.get("matched"))

    flag_fn = total_gt_of_cat - flag_tp

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
    loose_recall = loose_tp / n_gt_pos if n_gt_pos else None
    flag_precision = flag_tp / (flag_tp + flag_fp) if (flag_tp + flag_fp) else None
    flag_recall = flag_tp / (flag_tp + flag_fn) if (flag_tp + flag_fn) else None

    assert tp + fn == n_gt_pos, f"category={category} threshold={threshold}: TP+FN={tp + fn} != n_gt_pos={n_gt_pos}"
    assert fp + tn == n_gt_neg, f"category={category} threshold={threshold}: FP+TN={fp + tn} != n_gt_neg={n_gt_neg}"
    assert tp <= loose_tp, f"category={category} threshold={threshold}: strict tp={tp} > loose_tp={loose_tp}"

    return {
        "category": category,
        "threshold": threshold if threshold is not None else "none",
        "n_gt_pos": n_gt_pos,
        "n_gt_neg": n_gt_neg,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": r2(precision), "recall": r2(recall),
        "specificity": r2(specificity), "accuracy": r2(accuracy), "f1": r2(f1),
        "loose_tp": loose_tp, "loose_recall": r2(loose_recall),
        "flag_tp": flag_tp, "flag_fp": flag_fp, "flag_fn": flag_fn,
        "flag_precision": r2(flag_precision), "flag_recall": r2(flag_recall),
        "n_flags_missing_confidence": n_flags_missing_confidence,
    }


def threshold_metrics_table(rows: List[dict], thresholds: List[float], logger: Optional[StageLogger] = None) -> List[dict]:
    """Builds every (category, threshold) row for one language-scoped `rows`
    list, including the "none" row, and runs the cross-threshold/consistency
    checks from the spec. Returns the rows (without model/language — caller
    adds those); raises AssertionError if a monotonicity/count check fails."""
    out = []
    for category in THRESHOLD_TABLE_CATEGORIES:
        group_rows = [threshold_metrics_for_group(rows, category, t) for t in thresholds] + \
                     [threshold_metrics_for_group(rows, category, None)]
        # Monotonicity as threshold rises (the "none" row excluded — it's
        # not part of the rising-threshold sequence, it's the unfiltered
        # baseline): TP/FP never increase, TN never decreases.
        numeric_rows = group_rows[:len(thresholds)]
        for prev, cur in zip(numeric_rows, numeric_rows[1:]):
            assert cur["tp"] <= prev["tp"], f"category={category}: TP rose from {prev['tp']} (t={prev['threshold']}) to {cur['tp']} (t={cur['threshold']})"
            assert cur["fp"] <= prev["fp"], f"category={category}: FP rose from {prev['fp']} (t={prev['threshold']}) to {cur['fp']} (t={cur['threshold']})"
            assert cur["tn"] >= prev["tn"], f"category={category}: TN fell from {prev['tn']} (t={prev['threshold']}) to {cur['tn']} (t={cur['threshold']})"
        out.extend(group_rows)

    # "none" + ALL must reproduce confusion_file_level_overall/by_language
    # and confusion_flag_level exactly.
    none_all = next(r for r in out if r["category"] == "ALL" and r["threshold"] == "none")
    expected_file = aggregate_file_level(rows)
    expected_flag = aggregate_flag_level(rows)
    mismatches = []
    for key in ("tp", "fp", "fn", "tn"):
        if none_all[key] != expected_file[key]:
            mismatches.append(f"file-level {key}: threshold-table={none_all[key]} vs confusion={expected_file[key]}")
    if none_all["flag_tp"] != expected_flag["tp"]:
        mismatches.append(f"flag_tp: threshold-table={none_all['flag_tp']} vs confusion={expected_flag['tp']}")
    if none_all["flag_fp"] != expected_flag["fp"]:
        mismatches.append(f"flag_fp: threshold-table={none_all['flag_fp']} vs confusion={expected_flag['fp']}")
    if none_all["flag_fn"] != expected_flag["fn"]:
        mismatches.append(f"flag_fn: threshold-table={none_all['flag_fn']} vs confusion={expected_flag['fn']}")
    if mismatches:
        msg = "threshold_metrics_table 'none'/ALL row disagrees with confusion_file_level/confusion_flag_level: " + "; ".join(mismatches)
        if logger:
            logger.warn(msg)
        else:
            print(f"WARNING: {msg}")
    return out


def load_cached_rows(output_dir: Path, model_key: str) -> List[dict]:
    """Reads analyze_results.py's ALREADY-WRITTEN per-file cache — no
    scoring, no matching, no API calls. A model that was never scored by
    analyze_results.py (no per_file/ dir) yields an empty list."""
    per_file_dir = output_dir / model_key / "per_file"
    rows = []
    for fp in sorted(glob.glob(str(per_file_dir / "*.json"))):
        rows.append(json.loads(Path(fp).read_text()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None,
                         help="Comma-separated subset of model keys; default = every model with an existing per_file/ cache")
    parser.add_argument("--dataset-root", type=str, default=None,
                         help="Alternate dataset root, e.g. Dostt_dev — same convention as analyze_results.py")
    parser.add_argument("--results-tag", type=str, default=None,
                         help="Extra results-directory suffix, e.g. promptA — same convention as analyze_results.py")
    parser.add_argument("--thresholds", type=str, default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                         help="Comma-separated model_confidence thresholds (default: 0.0,0.1,...,0.9). "
                              "A 'none' row (every flag, missing confidence included) is always added on top.")
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    configure_dataset_root(args.dataset_root, args.results_tag)
    output_dir = ar.OUTPUT_DIR
    logger = StageLogger("threshold_analysis")

    if args.models:
        model_keys = args.models.split(",")
    else:
        model_keys = sorted(p.parent.name for p in output_dir.glob("*/per_file") if p.is_dir())

    if not model_keys:
        logger.warn(f"No per_file/ caches found under {output_dir}/ — run analyze_results.py for this "
                    f"dataset-root/results-tag first (this script never scores files itself).")
        return

    threshold_rows = []
    for model_key in model_keys:
        rows = load_cached_rows(output_dir, model_key)
        if not rows:
            logger.warn(f"[{model_key}] no cached per_file rows found — skipping")
            continue

        for row in threshold_metrics_table(rows, thresholds, logger):
            threshold_rows.append({"model": model_key, "language": "ALL", **row})

        by_lang: dict = {}
        for r in rows:
            by_lang.setdefault(r["language"], []).append(r)
        for lang, lang_rows in sorted(by_lang.items()):
            for row in threshold_metrics_table(lang_rows, thresholds, logger):
                threshold_rows.append({"model": model_key, "language": lang, **row})

        logger.info(f"[{model_key}] {len(rows)} cached file(s) scored into threshold_metrics_by_category.csv")

    merge_and_write_csv(
        output_dir / "threshold_metrics_by_category.csv", threshold_rows,
        ["model", "language", "category", "threshold", "n_gt_pos", "n_gt_neg", "tp", "fp", "fn", "tn",
         "precision", "recall", "specificity", "accuracy", "f1", "loose_tp", "loose_recall",
         "flag_tp", "flag_fp", "flag_fn", "flag_precision", "flag_recall", "n_flags_missing_confidence"],
        model_keys,
    )
    print(f"threshold_metrics_by_category.csv written to {output_dir}/ for {len(model_keys)} model(s) — "
          f"no other file in that directory was touched.")


if __name__ == "__main__":
    main()
