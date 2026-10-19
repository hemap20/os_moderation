"""Full-dataset leaderboard: one row per (prompt, model) run, rebuilt from
every scored full-dataset analysis folder each time it runs. analyze_results.py
calls it at the end of every full-dataset run, so it stays up to date; it can
also be run by hand.

Basis (same for every row, so trends are comparable):
  - confidence: pp_full (scoring_modes.adjusted_confidence) for v6+ prompts.
    v1 has none of the fields pp_full uses, so v1 rows use raw confidence
    (pp_full would only turn missing confidence into 0.0 there).
  - operating point: confidence >= 0.8 (HIGH). For v6-v8, >=0.8 and >=0.9
    are identical: those prompts only output 0.1 / 0.3 / 0.6 / 0.9.
  - per call (file level) metrics reuse threshold_analysis.threshold_metrics_for_group,
    per flag metrics reuse threshold_analysis's flag rules, so every number
    matches the existing threshold / bucket tables.

Columns:
  recall, specificity, balanced_acc   per call, all categories, conf >= 0.8.
                                       balanced_acc = (recall + specificity) / 2.
  call_precision                       per call, NO threshold (every flag counts): of the calls
                                       flagged, the share that really had a violation caught
                                       (= the "overall" row of the Overall table).
  flag_precision                       per flag, NO threshold: share of all flags that are
                                       correct (matched, or a redundant extra catch of an
                                       already-matched violation).
  precision_gap                        flag precision at conf >= 0.8 minus flag
                                       precision below 0.8. Larger = confidence is
                                       more meaningful; 0 = confidence carries no signal.
  right_flags_high                     share of all correct flags that have conf >= 0.8.
                                       Higher = the model puts its correct flags at
                                       high confidence (so a threshold keeps them).
  ambiguous_recall                     per call recall at conf >= 0.8 on violation calls
                                       whose ground-truth confidence is < 0.7 (the
                                       "ambiguous_violation" slice of audio_segments.csv).

Usage: python3 leaderboard.py
Writes analysis_results_full_summary/leaderboard.csv and leaderboard.md.
"""
import csv
import glob
import json
import re
from datetime import date
from pathlib import Path

import config
import scoring_modes as sm
from threshold_analysis import _is_redundant, threshold_metrics_for_group

OUT_DIR = config.PROJECT_ROOT / "analysis_results_full_summary"
SEGMENTS_CSV = OUT_DIR / "audio_segments.csv"
HIGH = 0.8
TOTAL_FILES = 381
SIGNAL = "_lb_conf"

COLUMNS = ["rank", "model", "input", "prompt", "calls", "partial", "basis",
           "balanced_acc", "recall", "specificity", "call_precision", "flag_precision", "precision_gap",
           "right_flags_high", "ambiguous_recall", "n_flags", "updated"]


def full_dataset_runs():
    """(prompt, folder) for every FULL-dataset analysis folder: analysis_results
    (v1) and analysis_results_promptvN. Dev-set folders (analysis_results_dev*)
    are excluded."""
    runs = []
    for d in sorted(config.PROJECT_ROOT.glob("analysis_results*")):
        if not d.is_dir():
            continue
        if d.name == "analysis_results":
            runs.append(("v1", d))
        else:
            m = re.fullmatch(r"analysis_results_promptv(\d+)", d.name)
            if m:
                runs.append((f"v{m.group(1)}", d))
    return runs


def model_input(model_key):
    if "audiotrans" in model_key:
        return "text (English, Gemini-translated from audio)"
    if "itc" in model_key:
        return "text (Indic-Transcribe-Core)"
    if "conformer" in model_key:
        return "text (Indic-Conformer)"
    return "audio"


def load_rows(folder: Path, model: str):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(folder / model / "per_file" / "*.json")))]


def scored_rows(rows, prompt):
    """Copies rows with each flag's leaderboard confidence under SIGNAL;
    pp_full drops (None) are removed. v1 keeps raw confidence."""
    use_pp = prompt != "v1"
    out = []
    for row in rows:
        flags = []
        for f in row.get("model_flags", []):
            conf = sm.adjusted_confidence(f, "pp_full", prompt) if use_pp else f.get("model_confidence")
            if use_pp and conf is None:
                continue  # dropped by pp_full (invalid category)
            nf = dict(f)
            nf[SIGNAL] = conf
            flags.append(nf)
        nr = dict(row)
        nr["model_flags"] = flags
        out.append(nr)
    return out


def r2(x):
    return round(x, 2) if x is not None else None


def flag_metrics(rows):
    hi_ok = hi_n = lo_ok = lo_n = 0
    for row in rows:
        for f in row["model_flags"]:
            ok = bool(f.get("matched")) or _is_redundant(f)
            c = f.get(SIGNAL)
            if c is not None and c >= HIGH:
                hi_n += 1
                hi_ok += ok
            else:
                lo_n += 1
                lo_ok += ok
    hi_p = hi_ok / hi_n if hi_n else None
    lo_p = lo_ok / lo_n if lo_n else None
    gap = (hi_p - (lo_p or 0.0)) if hi_p is not None else None
    right = hi_ok + lo_ok
    return {
        "flag_precision": r2(right / (hi_n + lo_n)) if (hi_n + lo_n) else None,  # all flags, no threshold
        "precision_gap": r2(gap),
        "right_flags_high": r2(hi_ok / right) if right else None,
        "n_flags": hi_n + lo_n,
    }


def build():
    ambiguous = set()
    if SEGMENTS_CSV.exists():
        with SEGMENTS_CSV.open() as fh:
            ambiguous = {r["file_id"] for r in csv.DictReader(fh) if r["difficulty_segment"] == "ambiguous_violation"}
    else:
        print(f"WARNING: {SEGMENTS_CSV} missing (run threshold_by_bucket_full.py) — ambiguous_recall left blank")

    rows_out = []
    for prompt, folder in full_dataset_runs():
        for model_dir in sorted(p.parent for p in folder.glob("*/per_file") if p.is_dir()):
            raw = load_rows(folder, model_dir.name)
            if not raw:
                continue
            rows = scored_rows(raw, prompt)
            m = threshold_metrics_for_group(rows, "ALL", HIGH, SIGNAL)
            rec, spec = m["recall"], m["specificity"]
            amb_rows = [r for r in rows if r["file_id"] in ambiguous]
            amb = threshold_metrics_for_group(amb_rows, "ALL", HIGH, SIGNAL)["recall"] if amb_rows else None
            rows_out.append({
                "model": model_dir.name, "input": model_input(model_dir.name), "prompt": prompt,
                "calls": len(rows), "partial": "yes" if len(rows) < 0.9 * TOTAL_FILES else "",
                "basis": "raw" if prompt == "v1" else "pp_full",
                "balanced_acc": r2((rec + spec) / 2) if rec is not None and spec is not None else None,
                "recall": rec, "specificity": spec, "call_precision": threshold_metrics_for_group(rows, "ALL", None, SIGNAL)["precision"], "ambiguous_recall": amb,
                **flag_metrics(rows), "updated": date.today().isoformat(),
            })
    rows_out.sort(key=lambda r: -(r["balanced_acc"] or 0))
    for i, r in enumerate(rows_out, 1):
        r["rank"] = i
    return rows_out


def write_md(rows, path):
    hdr = ["#", "Model", "Input", "Prompt", "Calls", "Balanced acc.", "Recall", "Specificity",
           "Precision per call (all flags)", "Precision per flag (all flags)", "Precision gap", "Right flags ≥0.8", "Ambiguous recall"]
    lines = ["# Full-dataset leaderboard", "",
             f"Per call at confidence ≥ {HIGH} unless noted (both precisions use all flags, no threshold); pp_full confidence (raw for v1). "
             "\\* = partial run (fewer calls scored). Regenerate: `python3 leaderboard.py`.", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        star = " \\*" if r["partial"] else ""
        cells = [r["rank"], r["model"] + star, r["input"], r["prompt"], r["calls"], r["balanced_acc"], r["recall"],
                 r["specificity"], r["call_precision"], r["flag_precision"], r["precision_gap"], r["right_flags_high"], r["ambiguous_recall"]]
        lines.append("| " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    rows = build()
    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "leaderboard.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    write_md(rows, OUT_DIR / "leaderboard.md")
    print(f"leaderboard: {len(rows)} runs -> {OUT_DIR / 'leaderboard.csv'} and leaderboard.md")


if __name__ == "__main__":
    main()
