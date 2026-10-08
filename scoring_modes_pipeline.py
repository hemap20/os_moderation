"""Scoring-modes pipeline: layers scoring_modes.py's four confidence modes
(raw, sa, sa_drop_invalid, pp_full) on top of analyze_results.py's already-
matched per-file cache, for every dev prompt-version folder, then builds the
cross-prompt comparison.

This is read-only over matching/ground-truth/raw-model-output — it only
ever ADDS a "scored" dict to each cached flag and a "scoring_modes_version"
file-level key, via the same safe index-free merge-by-identity pattern used
throughout this project's earlier backfills (never touches "matched",
"matched_gt_category", "fp_classification", or any existing field).

Threshold tables are produced by calling threshold_analysis.py's EXISTING
threshold_metrics_table/threshold_metrics_for_group unchanged, once per
scoring mode, via a throwaway "_scored_signal" key written onto a COPY of
each flag dict for that mode's pass — no second threshold implementation.

Usage:
    python3 scoring_modes_pipeline.py --dataset-root Dostt_dev --results-tag promptv6 --prompt-version v6 --matcher-version mv1
    python3 scoring_modes_pipeline.py --cross-summary   # after per-folder runs, builds the cross-prompt files
"""
import argparse
import glob
import json
import math
from pathlib import Path
from typing import List, Optional

import config
import analyze_results as ar
from analyze_results import configure_dataset_root, r2, merge_and_write_csv
from threshold_analysis import (
    threshold_metrics_table,
    THRESHOLD_TABLE_CATEGORIES,
)
import scoring_modes as sm

MATCHER_VERSION = "mv1"  # the global_matcher.py design locked in this session (Hungarian + embeddings + gemini-3.8-flash verify)

ALL_MODES_THRESHOLDS = [round(i * 0.1, 1) for i in range(10)] + [0.95]
DROP_MODES = {"sa_drop_invalid", "pp_full"}

DEV_FOLDERS = {
    "v1": ("analysis_results_dev_v1prompt", None, False),
    "v2": ("analysis_results_dev_promptv2", None, False),
    "v3": ("analysis_results_dev_promptv3", None, False),
    "v4": ("analysis_results_dev_promptv4", None, False),
    "v5": ("analysis_results_dev_promptv5", None, True),  # schema_mismatch
    "v6": ("analysis_results_dev_promptv6", None, False),
    "v7": ("analysis_results_dev_promptv7", None, False),
    "v8": ("analysis_results_dev_promptv8", None, False),
}


def _models_present(folder: Path) -> List[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and (p / "per_file").is_dir())


def _load_rows(folder: Path, model: str) -> List[dict]:
    per_file_dir = folder / model / "per_file"
    rows = []
    for fp in sorted(glob.glob(str(per_file_dir / "*.json"))):
        rows.append((Path(fp), json.loads(Path(fp).read_text())))
    return rows


def _persist_scored(path_rows: List[tuple]):
    for fp, row in path_rows:
        row["scoring_modes_version"] = sm.SCORING_MODES_VERSION
        fp.write_text(json.dumps(row, indent=2, ensure_ascii=False))


def compute_scored(folder: Path, model: str, prompt_version: str) -> List[dict]:
    """Loads cached rows, attaches flag['scored'] = {mode: value} to every
    model flag (mutating in place), persists back to disk, returns the rows
    (list of dicts, path stripped) for downstream use."""
    path_rows = _load_rows(folder, model)
    for fp, row in path_rows:
        for flag in row.get("model_flags", []):
            flag["scored"] = sm.score_flag_all_modes(flag, prompt_version)
    _persist_scored(path_rows)
    return [row for _, row in path_rows]


def _rows_for_mode(rows: List[dict], mode: str) -> List[dict]:
    """Returns a shallow-copied rows structure where each KEPT flag carries
    a '_scored_signal' key = its adjusted confidence for `mode`, and dropped
    flags (sa_drop_invalid/pp_full, category invalid) are removed entirely —
    so even the 'none' threshold row excludes them, per spec."""
    out_rows = []
    for row in rows:
        new_flags = []
        for flag in row.get("model_flags", []):
            value = flag["scored"][mode]
            if value is None and mode in DROP_MODES:
                continue  # dropped
            new_flag = dict(flag)
            new_flag["_scored_signal"] = value
            new_flags.append(new_flag)
        new_row = dict(row)
        new_row["model_flags"] = new_flags
        out_rows.append(new_row)
    return out_rows


def _n_kept_dropped(rows: List[dict], mode: str, category: str):
    kept = dropped = 0
    for row in rows:
        for flag in row.get("model_flags", []):
            if category != "ALL" and flag.get("category") != category:
                continue
            if flag["scored"][mode] is None and mode in DROP_MODES:
                dropped += 1
            else:
                kept += 1
    return kept, dropped


def _redundant_available(rows: List[dict], category: str) -> bool:
    for row in rows:
        for flag in row.get("model_flags", []):
            if category != "ALL" and flag.get("category") != category:
                continue
            if flag.get("fp_classification"):
                return True
    return False


def build_all_modes_table(rows: List[dict], model: str, prompt_version: str, schema_mismatch: bool) -> List[dict]:
    out = []
    for mode in sm.SCORING_MODES:
        mode_rows = _rows_for_mode(rows, mode)
        table = threshold_metrics_table(mode_rows, ALL_MODES_THRESHOLDS, signal_key="_scored_signal")
        for category in THRESHOLD_TABLE_CATEGORIES:
            kept, dropped = _n_kept_dropped(rows, mode, category)
            redundant_avail = _redundant_available(rows, category)
            for r in table:
                if r["category"] != category:
                    continue
                r2d = dict(r)
                r2d["scoring_mode"] = mode
                r2d["scoring_modes_version"] = sm.SCORING_MODES_VERSION
                r2d["model"] = model
                r2d["prompt_version"] = prompt_version
                r2d["matcher_version"] = MATCHER_VERSION
                r2d["language"] = "ALL"
                r2d["strict_tp"] = r2d["tp"]
                r2d["strict_recall"] = r2d["recall"]
                r2d["n_flags_kept"] = kept
                r2d["n_flags_dropped"] = dropped
                r2d["redundant_available"] = redundant_avail
                if not redundant_avail:
                    r2d["flag_redundant"] = ""
                out.append(r2d)
        # per-language rows
        for lang in config.LANGUAGES:
            lang_rows = [r for r in mode_rows if r.get("language") == lang]
            if not lang_rows:
                continue
            lang_table = threshold_metrics_table(lang_rows, ALL_MODES_THRESHOLDS, signal_key="_scored_signal")
            raw_lang_rows = [r for r in rows if r.get("language") == lang]
            for category in THRESHOLD_TABLE_CATEGORIES:
                kept, dropped = _n_kept_dropped(raw_lang_rows, mode, category)
                redundant_avail = _redundant_available(raw_lang_rows, category)
                for r in lang_table:
                    if r["category"] != category:
                        continue
                    r2d = dict(r)
                    r2d["scoring_mode"] = mode
                    r2d["scoring_modes_version"] = sm.SCORING_MODES_VERSION
                    r2d["model"] = model
                    r2d["prompt_version"] = prompt_version
                    r2d["matcher_version"] = MATCHER_VERSION
                    r2d["language"] = lang
                    r2d["strict_tp"] = r2d["tp"]
                    r2d["strict_recall"] = r2d["recall"]
                    r2d["n_flags_kept"] = kept
                    r2d["n_flags_dropped"] = dropped
                    r2d["redundant_available"] = redundant_avail
                    if not redundant_avail:
                        r2d["flag_redundant"] = ""
                    out.append(r2d)
    return out


ALL_MODES_FIELDNAMES = [
    "scoring_mode", "scoring_modes_version", "model", "prompt_version", "matcher_version", "language", "category",
    "threshold", "n_gt_pos", "n_gt_neg", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy",
    "f1", "strict_tp", "strict_recall", "loose_tp", "loose_recall", "flag_tp", "flag_fp", "flag_fn",
    "flag_redundant", "flag_precision", "flag_recall", "n_flags_kept", "n_flags_dropped",
    "n_flags_missing_confidence", "redundant_available",
]


def _auroc(pairs: List[tuple]) -> Optional[float]:
    """pairs: list of (score, is_match). Mann-Whitney U based AUROC."""
    pos = [s for s, m in pairs if m]
    neg = [s for s, m in pairs if not m]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda x: x[0])
    ranks = {}
    i = 0
    n = len(ranked)
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
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def scoring_mode_diagnostics(rows: List[dict], model: str, prompt_version: str) -> List[dict]:
    out = []
    for mode in sm.SCORING_MODES:
        n_total = n_dropped = n_dropped_tp = n_capped = n_capped_tp = 0
        n_pushed_06 = n_pushed_08 = 0
        auroc_pairs = []
        auroc_pairs_dropped_lowest = []
        sa_by_act = {}
        sa_by_speaker = {}
        for row in rows:
            for flag in row.get("model_flags", []):
                n_total += 1
                raw_val = flag["scored"]["raw"]
                mode_val = flag["scored"][mode]
                matched = bool(flag.get("matched"))
                dropped = mode_val is None and mode in DROP_MODES
                if dropped:
                    n_dropped += 1
                    if matched:
                        n_dropped_tp += 1
                    auroc_pairs_dropped_lowest.append((-1.0, matched))
                    continue
                if raw_val is not None and mode_val is not None and mode_val < raw_val:
                    n_capped += 1
                    if matched:
                        n_capped_tp += 1
                if matched and raw_val is not None and mode_val is not None:
                    if raw_val >= 0.6 and mode_val < 0.6:
                        n_pushed_06 += 1
                    if raw_val >= 0.8 and mode_val < 0.8:
                        n_pushed_08 += 1
                sig = mode_val if mode_val is not None else 0.0
                auroc_pairs.append((sig, matched))
                auroc_pairs_dropped_lowest.append((sig, matched))
                if mode in ("sa", "pp_full"):
                    act = flag.get("model_speech_act") or "(missing)"
                    sa_by_act.setdefault(act, [0, 0])
                    sa_by_act[act][1] += 1
                    if matched:
                        sa_by_act[act][0] += 1
                    spk = flag.get("model_speaker") or "(missing)"
                    sa_by_speaker.setdefault(spk, [0, 0])
                    sa_by_speaker[spk][1] += 1
                    if matched:
                        sa_by_speaker[spk][0] += 1
        row_out = {
            "model": model, "prompt_version": prompt_version, "scoring_mode": mode,
            "n_flags_total": n_total, "n_dropped": n_dropped, "n_dropped_tp": n_dropped_tp,
            "n_capped": n_capped, "n_capped_tp": n_capped_tp,
            "n_tp_pushed_below_0.6": n_pushed_06, "n_tp_pushed_below_0.8": n_pushed_08,
            "auroc_confidence": r2(_auroc(auroc_pairs)),
            "auroc_with_dropped_as_lowest": r2(_auroc(auroc_pairs_dropped_lowest)),
            "tp_rate_by_speech_act": json.dumps({k: r2(v[0] / v[1]) for k, v in sa_by_act.items()}) if sa_by_act else "",
            "tp_rate_by_speaker": json.dumps({k: r2(v[0] / v[1]) for k, v in sa_by_speaker.items()}) if sa_by_speaker else "",
        }
        out.append(row_out)
    return out


def check_sa_equals_raw_for_v1_3(rows: List[dict], prompt_version: str):
    if prompt_version not in ("v1", "v2", "v3"):
        return
    for row in rows:
        for flag in row.get("model_flags", []):
            raw_v = flag["scored"]["raw"]
            sa_v = flag["scored"]["sa"]
            expected = 0.0 if raw_v is None else raw_v
            assert sa_v == expected, (
                f"{prompt_version}: sa != raw (missing->0.0) for a flag in {row.get('file_id')}: "
                f"raw={raw_v} sa={sa_v}"
            )


def run_folder(dataset_root: Optional[str], results_tag: Optional[str], prompt_version: str):
    configure_dataset_root(dataset_root, results_tag)
    folder = Path(ar.OUTPUT_DIR)
    models = _models_present(folder)
    print(f"[{prompt_version}] folder={folder} models={models}")
    all_mode_rows = []
    diag_rows = []
    for model in models:
        rows = compute_scored(folder, model, prompt_version)
        check_sa_equals_raw_for_v1_3(rows, prompt_version)
        schema_mismatch = DEV_FOLDERS.get(prompt_version, (None, None, False))[2]
        all_mode_rows.extend(build_all_modes_table(rows, model, prompt_version, schema_mismatch))
        diag_rows.extend(scoring_mode_diagnostics(rows, model, prompt_version))
        # check 2: raw mode / none / ALL reproduces existing confusion tables —
        # threshold_metrics_table already asserts/warns on this internally.
        print(f"  {model}: {len(rows)} files scored, {sum(len(r['model_flags']) for r in rows)} flags")
    merge_and_write_csv(folder / "threshold_metrics_by_category_all_modes.csv", all_mode_rows,
                         ALL_MODES_FIELDNAMES, models)
    merge_and_write_csv(folder / "scoring_mode_diagnostics.csv", diag_rows,
                         ["model", "prompt_version", "scoring_mode", "n_flags_total", "n_dropped", "n_dropped_tp",
                          "n_capped", "n_capped_tp", "n_tp_pushed_below_0.6", "n_tp_pushed_below_0.8",
                          "auroc_confidence", "auroc_with_dropped_as_lowest", "tp_rate_by_speech_act",
                          "tp_rate_by_speaker"], models)
    print(f"[{prompt_version}] wrote threshold_metrics_by_category_all_modes.csv ({len(all_mode_rows)} rows) "
          f"and scoring_mode_diagnostics.csv ({len(diag_rows)} rows)")
    return all_mode_rows


def build_cross_prompt_summary():
    out_dir = Path("analysis_results_dev_summary")
    out_dir.mkdir(exist_ok=True)
    all_rows = []
    matcher_versions_by_pv = {}
    for pv, (folder_name, _, schema_mismatch) in DEV_FOLDERS.items():
        folder = Path(folder_name)
        fp = folder / "threshold_metrics_by_category_all_modes.csv"
        if not fp.exists():
            print(f"skip {pv}: {fp} not found (run_folder first)")
            continue
        import csv
        with fp.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["schema_mismatch"] = schema_mismatch
                all_rows.append(r)
                matcher_versions_by_pv.setdefault(pv, set()).add(r["matcher_version"])

    # check 6: consistent matcher_version across compared prompt versions
    bad = {pv: v for pv, v in matcher_versions_by_pv.items() if len(v) > 1}
    if bad:
        raise AssertionError(f"Inconsistent matcher_version within a prompt version's own results: {bad}")
    distinct = {pv: next(iter(v)) for pv, v in matcher_versions_by_pv.items()}
    need_rescoring = [pv for pv, v in distinct.items() if v != MATCHER_VERSION]
    if need_rescoring:
        print(f"WARNING: these prompt versions were not scored with the current matcher ({MATCHER_VERSION}): {need_rescoring}")

    summary_rows = []
    for r in all_rows:
        if r["category"] != "ALL" or r["language"] != "ALL":
            continue
        if r["threshold"] not in ("none", "0.6", "0.8", "0.9"):
            continue
        summary_rows.append({
            "model": r["model"], "prompt_version": r["prompt_version"], "scoring_mode": r["scoring_mode"],
            "threshold": r["threshold"], "recall": r["recall"], "specificity": r["specificity"],
            "precision": r["precision"], "flag_precision": r["flag_precision"],
            "fp_files": r["fp"], "fn_files": r["fn"],
            "n_flags_at_or_above": r["flag_tp"] + r["flag_fp"] if r["flag_tp"] != "" and r["flag_fp"] != "" else "",
        })
    merge_and_write_csv(out_dir / "prompt_comparison_by_mode.csv", summary_rows,
                         ["model", "prompt_version", "scoring_mode", "threshold", "recall", "specificity",
                          "precision", "flag_precision", "fp_files", "fn_files", "n_flags_at_or_above"],
                         list({r["model"] for r in summary_rows}))

    # best_prompt_by_model.md
    md_lines = ["# Best prompt version + scoring mode, by model", "",
                "Differences of 1-2 files are within dev-set noise on this dataset size.", ""]
    models = sorted({r["model"] for r in summary_rows})
    for model in models:
        md_lines.append(f"## {model}")
        for thr in ("0.6", "0.8", "0.9"):
            candidates = [r for r in summary_rows if r["model"] == model and r["threshold"] == thr
                          and r["recall"] not in ("", None)]
            def sort_key(r):
                return (-float(r["recall"]), -(float(r["specificity"]) if r["specificity"] not in ("", None) else -1),
                        -(float(r["flag_precision"]) if r["flag_precision"] not in ("", None) else -1))
            candidates.sort(key=sort_key)
            md_lines.append(f"### threshold >= {thr}")
            for c in candidates[:3]:
                n_pos = int(c["recall"] == c["recall"]) and None
                md_lines.append(
                    f"- **{c['prompt_version']} / {c['scoring_mode']}** — recall {c['recall']}, "
                    f"specificity {c['specificity']}, flag_precision {c['flag_precision']} "
                    f"(fp_files={c['fp_files']}, fn_files={c['fn_files']})"
                )
            md_lines.append("")
    (out_dir / "best_prompt_by_model.md").write_text("\n".join(md_lines))
    print(f"wrote {out_dir / 'prompt_comparison_by_mode.csv'} ({len(summary_rows)} rows) "
          f"and {out_dir / 'best_prompt_by_model.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--results-tag", type=str, default=None)
    parser.add_argument("--prompt-version", type=str, default=None, choices=list(sm.PROMPT_VERSION_TO_FILE))
    parser.add_argument("--cross-summary", action="store_true")
    args = parser.parse_args()
    if args.cross_summary:
        build_cross_prompt_summary()
        return
    if not args.prompt_version:
        parser.error("--prompt-version is required unless --cross-summary")
    run_folder(args.dataset_root, args.results_tag, args.prompt_version)


if __name__ == "__main__":
    main()
