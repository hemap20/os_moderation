"""Threshold-wise post-processed metrics for prompts v6/v7/v8 (dev set),
reusing scoring_modes.py (modes) and threshold_analysis.py's existing
_passes_threshold/_signal_value/threshold_metrics_for_group (metric
definitions) — this script only adds per-file-id tracking (fn_files/
fp_files), the wide-format overview, mode diagnostics, and the confidence
distribution, none of which touch the metric definitions themselves.

Usage: python3 threshold_pp_v6_v8.py
"""
import csv
import glob
import json
from pathlib import Path

import config
import analyze_results as ar
from analyze_results import configure_dataset_root, r2, merge_and_write_csv
import threshold_analysis as ta
import scoring_modes as sm
from scoring_modes_pipeline import MATCHER_VERSION, DROP_MODES

DEV_FOLDERS = {
    "v6": "analysis_results_dev_promptv6",
    "v7": "analysis_results_dev_promptv7",
    "v8": "analysis_results_dev_promptv8",
}
FULL_FOLDERS = {
    "v6": "analysis_results_promptv6",
}
MODES = ["raw", "sa", "sa_drop_invalid", "pp_full"]
NUMERIC_THRESHOLDS = [round(i * 0.1, 2) for i in range(10)] + [0.95, 1.0]
ALL_THRESHOLDS = NUMERIC_THRESHOLDS + [None]
CATEGORIES = list(config.CATEGORY_LABELS.values()) + ["ALL"]

import sys
_FULL = "--full" in sys.argv
FOLDERS = FULL_FOLDERS if _FULL else DEV_FOLDERS
OUT_DIR = Path("analysis_results_full_summary/threshold_pp_v6") if _FULL else \
    Path("analysis_results_dev_summary/threshold_pp_v6_v8")


def short_id(file_id: str) -> str:
    return file_id.split("_")[2] if file_id.count("_") >= 2 else file_id


def load_rows(folder: str, model: str):
    per_file_dir = Path(folder) / model / "per_file"
    rows = []
    for fp in sorted(glob.glob(str(per_file_dir / "*.json"))):
        rows.append(json.loads(Path(fp).read_text()))
    return rows


def models_present(folder: str):
    return sorted(p.name for p in Path(folder).iterdir() if p.is_dir() and (p / "per_file").is_dir())


def ensure_scored(rows, prompt_version):
    for row in rows:
        for flag in row.get("model_flags", []):
            if "scored" not in flag:
                flag["scored"] = sm.score_flag_all_modes(flag, prompt_version)


# ---------------------------------------------------------------- step 1 ---

def preflight(folder: str, model: str, rows, prompt_version: str):
    n_files = len(rows)
    n_flags = sum(len(r["model_flags"]) for r in rows)
    missing = {"model_confidence": 0, "model_speech_act": 0, "model_speaker": 0, "model_violation": 0}
    n_invalid_cat = 0
    for r in rows:
        for f in r["model_flags"]:
            for k in missing:
                if f.get(k) is None:
                    missing[k] += 1
            if f.get("category") not in sm.VALID_CATEGORIES:
                n_invalid_cat += 1
    rates = {k: r2(v / n_flags) if n_flags else None for k, v in missing.items()}
    expected = 381 if _FULL else 50
    print(f"  {model}: {n_files} files (expected {expected}){' <-- MISMATCH' if n_files != expected else ''}, "
          f"{n_flags} flags, invalid_category={n_invalid_cat}")
    print(f"    missing rates: {rates}")
    return {"n_files": n_files, "n_flags": n_flags, "missing_rates": rates, "n_invalid_category": n_invalid_cat}


# ---------------------------------------------------------- mode row prep --

def rows_for_mode(rows, mode):
    """Mirrors scoring_modes_pipeline._rows_for_mode: drops flags whose
    scored[mode] is None in a drop mode, tags kept flags with a plain
    '_scored_signal' key so threshold_analysis's EXISTING engine can be
    reused unchanged via its generic flag.get(signal_key) fallback."""
    out = []
    for row in rows:
        new_flags = []
        for flag in row.get("model_flags", []):
            value = flag["scored"][mode]
            if value is None and mode in DROP_MODES:
                continue
            nf = dict(flag)
            nf["_scored_signal"] = value
            new_flags.append(nf)
        nr = dict(row)
        nr["model_flags"] = new_flags
        out.append(nr)
    return out


def file_level_ids(rows, category, threshold, signal_key="_scored_signal"):
    """Mirrors threshold_metrics_for_group's own file-level TP/FP/FN/TN
    rule exactly (same _passes_threshold/_signal_value calls), just also
    returning which file IDs landed in fp/fn — threshold_metrics_for_group
    itself only returns counts."""
    fp_files, fn_files = [], []
    for r in rows:
        gt_flags = r.get("gt_flags", [])
        model_flags = r["model_flags"]
        if category == "ALL":
            gt_of_cat = gt_flags
        else:
            gt_of_cat = [g for g in gt_flags if g.get("category") == category]
        gt_pos = len(gt_of_cat) > 0

        strict_pool = [f for f in model_flags if ta._passes_threshold(f, threshold, signal_key)]
        if category == "ALL":
            model_of_cat = [f for f in model_flags if ta._passes_threshold(f, threshold, signal_key)]
            strict_pos = any(f.get("matched") for f in strict_pool)
        else:
            model_of_cat = [f for f in model_flags if f.get("category") == category
                             and ta._passes_threshold(f, threshold, signal_key)]
            strict_pos = any(f.get("matched") and f.get("matched_gt_category") == category for f in strict_pool)
        fp_pos = len(model_of_cat) > 0

        if gt_pos and not strict_pos:
            fn_files.append(short_id(r["file_id"]))
        elif not gt_pos and fp_pos:
            fp_files.append(short_id(r["file_id"]))
    return fp_files, fn_files


def n_flags_at_or_above(rows, category, threshold, signal_key="_scored_signal"):
    n = 0
    for r in rows:
        for f in r["model_flags"]:
            if category != "ALL" and f.get("category") != category:
                continue
            if ta._passes_threshold(f, threshold, signal_key):
                n += 1
    return n


def redundant_available(rows, category):
    for r in rows:
        for f in r["model_flags"]:
            if category != "ALL" and f.get("category") != category:
                continue
            if f.get("fp_classification"):
                return True
    return False


def n_kept_dropped(rows, mode, category):
    kept = dropped = 0
    for r in rows:
        for f in r["model_flags"]:
            if category != "ALL" and f.get("category") != category:
                continue
            if f["scored"][mode] is None and mode in DROP_MODES:
                dropped += 1
            else:
                kept += 1
    return kept, dropped


ALL_MODES_FIELDNAMES = [
    "model", "prompt_version", "scoring_mode", "matcher_version", "language", "category", "threshold",
    "n_gt_pos", "n_gt_neg", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy", "f1",
    "fp_files", "fn_files",
    "flag_tp", "flag_fp", "flag_fn", "flag_redundant", "flag_precision", "flag_recall", "n_flags_at_or_above",
    "n_flags_kept", "n_flags_dropped", "redundant_available",
]


def build_table(rows, model, prompt_version, language):
    out = []
    for mode in MODES:
        mode_rows = rows_for_mode(rows, mode)
        table = ta.threshold_metrics_table(mode_rows, NUMERIC_THRESHOLDS, signal_key="_scored_signal")
        for category in CATEGORIES:
            kept, dropped = n_kept_dropped(rows, mode, category)
            r_avail = redundant_available(rows, category)
            for row in table:
                if row["category"] != category:
                    continue
                thr_val = None if row["threshold"] == "none" else row["threshold"]
                fp_files, fn_files = file_level_ids(mode_rows, category, thr_val)
                n_at = n_flags_at_or_above(mode_rows, category, thr_val)
                out_row = dict(row)
                out_row.update({
                    "model": model, "prompt_version": prompt_version, "scoring_mode": mode,
                    "matcher_version": MATCHER_VERSION, "language": language,
                    "fp_files": ";".join(fp_files), "fn_files": ";".join(fn_files),
                    "n_flags_at_or_above": n_at,
                    "n_flags_kept": kept, "n_flags_dropped": dropped, "redundant_available": r_avail,
                })
                if not r_avail:
                    out_row["flag_redundant"] = ""
                out.append(out_row)
    return out


# ---------------------------------------------------------- diagnostics ---

def auroc(pairs):
    pos = [s for s, m in pairs if m]
    neg = [s for s, m in pairs if not m]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda x: x[0])
    ranks = {}
    i, n = 0, len(ranked)
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[id(ranked[k])] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[id(p)] for p in ranked if p[1])
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mode_diagnostics(rows, model, prompt_version):
    out = []
    for mode in MODES:
        n_capped = n_capped_tp = n_dropped = n_dropped_tp = 0
        n_pushed_06 = n_pushed_08 = 0
        pairs = []
        sa_by_act, sa_by_speaker = {}, {}
        for row in rows:
            for f in row["model_flags"]:
                raw_v = f["scored"]["raw"]
                mode_v = f["scored"][mode]
                matched = bool(f.get("matched"))
                dropped = mode_v is None and mode in DROP_MODES
                if dropped:
                    n_dropped += 1
                    if matched:
                        n_dropped_tp += 1
                    pairs.append((-1.0, matched))
                    continue
                if raw_v is not None and mode_v is not None and mode_v < raw_v:
                    n_capped += 1
                    if matched:
                        n_capped_tp += 1
                if matched and raw_v is not None and mode_v is not None:
                    if raw_v >= 0.6 and mode_v < 0.6:
                        n_pushed_06 += 1
                    if raw_v >= 0.8 and mode_v < 0.8:
                        n_pushed_08 += 1
                pairs.append((mode_v if mode_v is not None else 0.0, matched))
                if mode in ("sa", "pp_full"):
                    act = f.get("model_speech_act") or "(missing)"
                    sa_by_act.setdefault(act, [0, 0]); sa_by_act[act][1] += 1
                    if matched: sa_by_act[act][0] += 1
                    spk = f.get("model_speaker") or "(missing)"
                    sa_by_speaker.setdefault(spk, [0, 0]); sa_by_speaker[spk][1] += 1
                    if matched: sa_by_speaker[spk][0] += 1
        out.append({
            "model": model, "prompt_version": prompt_version, "scoring_mode": mode,
            "n_capped": n_capped, "n_capped_tp": n_capped_tp,
            "n_dropped": n_dropped, "n_dropped_tp": n_dropped_tp,
            "tp_pushed_below_0.6": n_pushed_06, "tp_pushed_below_0.8": n_pushed_08,
            "auroc_confidence": r2(auroc(pairs)),
            "tp_rate_by_speech_act": json.dumps({k: r2(v[0] / v[1]) for k, v in sa_by_act.items()}) if sa_by_act else "",
            "tp_rate_by_speaker": json.dumps({k: r2(v[0] / v[1]) for k, v in sa_by_speaker.items()}) if sa_by_speaker else "",
        })
    return out


def confidence_distribution(rows, model, prompt_version):
    out = []
    for mode in MODES:
        buckets = {}
        for row in rows:
            for f in row["model_flags"]:
                v = f["scored"][mode]
                if v is None:
                    continue
                matched = bool(f.get("matched"))
                key = round(v, 3)
                d = buckets.setdefault(key, {"matched": 0, "unmatched": 0})
                d["matched" if matched else "unmatched"] += 1
        for val, d in sorted(buckets.items()):
            out.append({"model": model, "prompt_version": prompt_version, "scoring_mode": mode,
                        "confidence_value": val, "n_matched": d["matched"], "n_unmatched": d["unmatched"]})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Step 1: preflight ===")
    print(f"matcher_version used for all three folders (fixed constant, same global_matcher.py locked this session): {MATCHER_VERSION}")

    all_rows_out, diag_rows_out, dist_rows_out = [], [], []
    per_model_rows = {}  # (pv, model) -> rows (for overview)

    for pv, folder in FOLDERS.items():
        print(f"\n[{pv}] {folder}")
        for model in models_present(folder):
            rows = load_rows(folder, model)
            ensure_scored(rows, pv)
            preflight(folder, model, rows, pv)
            per_model_rows[(pv, model)] = rows

            all_rows_out.extend(build_table(rows, model, pv, "ALL"))
            for lang in config.LANGUAGES:
                lang_rows = [r for r in rows if r.get("language") == lang]
                if lang_rows:
                    all_rows_out.extend(build_table(lang_rows, model, pv, lang))

            diag_rows_out.extend(mode_diagnostics(rows, model, pv))
            dist_rows_out.extend(confidence_distribution(rows, model, pv))

            # check 5 (v1-3 only) n/a here; check 3 (kept+dropped==total) spot check
            for mode in MODES:
                kept, dropped = n_kept_dropped(rows, mode, "ALL")
                total = sum(len(r["model_flags"]) for r in rows)
                assert kept + dropped == total, f"{pv}/{model}/{mode}: kept+dropped != total"

    all_rows_out = [{k: r.get(k, "") for k in ALL_MODES_FIELDNAMES} for r in all_rows_out]
    merge_and_write_csv(OUT_DIR / "threshold_metrics_v6_v8_all_modes.csv", all_rows_out, ALL_MODES_FIELDNAMES,
                         list({r["model"] for r in all_rows_out}))
    merge_and_write_csv(OUT_DIR / "mode_diagnostics.csv", diag_rows_out,
                         ["model", "prompt_version", "scoring_mode", "n_capped", "n_capped_tp", "n_dropped",
                          "n_dropped_tp", "tp_pushed_below_0.6", "tp_pushed_below_0.8", "auroc_confidence",
                          "tp_rate_by_speech_act", "tp_rate_by_speaker"],
                         list({r["model"] for r in diag_rows_out}))
    merge_and_write_csv(OUT_DIR / "confidence_value_distribution.csv", dist_rows_out,
                         ["model", "prompt_version", "scoring_mode", "confidence_value", "n_matched", "n_unmatched"],
                         list({r["model"] for r in dist_rows_out}))
    print(f"\nwrote {len(all_rows_out)} rows to threshold_metrics_v6_v8_all_modes.csv")

    # --- overview_ALL.csv / .md (language ALL, category ALL only) ---
    overview_rows = [r for r in all_rows_out if r["language"] == "ALL" and r["category"] == "ALL"]
    OVERVIEW_THRESHOLDS_MD = [0.0, 0.3, 0.6, 0.8, 0.9, 1.0]

    wide_fieldnames = ["model", "prompt_version", "scoring_mode"]
    for t in NUMERIC_THRESHOLDS:
        for suffix in ("R", "Spec", "P", "flagP", "FPfiles", "FNfiles"):
            wide_fieldnames.append(f"t{t}_{suffix}")
    wide_rows = []
    models = sorted({r["model"] for r in overview_rows})
    for model in models:
        for pv in FOLDERS:
            for mode in MODES:
                cands = [r for r in overview_rows if r["model"] == model and r["prompt_version"] == pv
                         and r["scoring_mode"] == mode]
                if not cands:
                    continue
                wrow = {"model": model, "prompt_version": pv, "scoring_mode": mode}
                for t in NUMERIC_THRESHOLDS:
                    match = next((r for r in cands if r["threshold"] == t), None)
                    if match:
                        wrow[f"t{t}_R"] = match["recall"]
                        wrow[f"t{t}_Spec"] = match["specificity"]
                        wrow[f"t{t}_P"] = match["precision"]
                        wrow[f"t{t}_flagP"] = match["flag_precision"]
                        wrow[f"t{t}_FPfiles"] = match["fp"]
                        wrow[f"t{t}_FNfiles"] = match["fn"]
                wide_rows.append(wrow)
    write_rows = [{k: r.get(k, "") for k in wide_fieldnames} for r in wide_rows]
    ar.write_csv(OUT_DIR / "overview_ALL.csv", write_rows, wide_fieldnames)

    # markdown
    dataset_label = "full dataset (381 files)" if _FULL else "dev set"
    md = [f"# Threshold overview — {', '.join(FOLDERS)} {dataset_label} (category ALL, language ALL)", ""]
    for model in models:
        md.append(f"## {model}")
        header = "| prompt/mode | " + " | ".join(f"t={t}" for t in OVERVIEW_THRESHOLDS_MD) + " |"
        sep = "|---" * (len(OVERVIEW_THRESHOLDS_MD) + 1) + "|"
        md.append(header)
        md.append(sep)
        for pv in FOLDERS:
            for mode in ("raw", "sa", "pp_full"):
                cands = [r for r in overview_rows if r["model"] == model and r["prompt_version"] == pv
                         and r["scoring_mode"] == mode]
                if not cands:
                    continue
                cells = []
                for t in OVERVIEW_THRESHOLDS_MD:
                    match = next((r for r in cands if r["threshold"] == t), None)
                    if match:
                        cells.append(f"{match['recall']}/{match['specificity']}")
                    else:
                        cells.append("")
                md.append(f"| {pv}/{mode} | " + " | ".join(cells) + " |")
        md.append("")
        for t in (0.6, 0.8, 0.9):
            cands = [r for r in overview_rows if r["model"] == model and r["threshold"] == t and r["recall"] != ""]
            def sk(r):
                rec = float(r["recall"]) if r["recall"] != "" else -1
                spec = float(r["specificity"]) if r["specificity"] != "" else -1
                fp_ = float(r["flag_precision"]) if r["flag_precision"] != "" else -1
                return (-rec, -spec, -fp_)
            cands.sort(key=sk)
            if cands:
                c = cands[0]
                n_pos = c["tp"] + c["fn"]
                n_neg = c["tn"] + c["fp"]
                md.append(f"- Best at t={t}: **{c['prompt_version']}/{c['scoring_mode']}** — "
                          f"recall {c['recall']} ({c['tp']}/{n_pos}), specificity {c['specificity']} "
                          f"({c['tn']}/{n_neg}), flag_precision {c['flag_precision']}")
        md.append("")
        md.append("_Differences of 1-2 files are within dev-set noise on this dataset size._")
        md.append("")
    (OUT_DIR / "overview_ALL.md").write_text("\n".join(md))
    print(f"wrote overview_ALL.csv ({len(wide_rows)} rows) and overview_ALL.md")

    # --- check 2: raw/none/ALL reproduces confusion tables exactly ---
    print("\n=== Check 2: raw/none/ALL vs confusion tables ===")
    for pv, folder in FOLDERS.items():
        for model in models_present(folder):
            rows = per_model_rows[(pv, model)]
            mode_rows = rows_for_mode(rows, "raw")
            expected_file = ta.aggregate_file_level(mode_rows)
            expected_flag = ta.aggregate_flag_level(mode_rows)
            none_row = next(r for r in all_rows_out if r["model"] == model and r["prompt_version"] == pv
                             and r["scoring_mode"] == "raw" and r["language"] == "ALL"
                             and r["category"] == "ALL" and r["threshold"] == "none")
            for k in ("tp", "fp", "fn", "tn"):
                assert none_row[k] == expected_file[k], f"{pv}/{model}: file-level {k} mismatch"
            assert none_row["flag_tp"] == expected_flag["tp"], f"{pv}/{model}: flag_tp mismatch"
            print(f"  {pv}/{model}: OK")

    print("\nAll checks passed (1 TP+FN=n_gt_pos/FP+TN=n_gt_neg and 4 monotonicity enforced inline by "
          "threshold_analysis.threshold_metrics_for_group/table; 3 kept+dropped==total verified above; "
          f"2 raw/none/ALL verified above; 5 same matcher_version={MATCHER_VERSION} used throughout).")


if __name__ == "__main__":
    main()
