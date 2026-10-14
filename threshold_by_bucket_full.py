"""Threshold-wise metrics by audio bucket, for every full-dataset run.

Three independent bucketing axes, computed ONCE (audio_segments.csv) so
every run is measured against identical groups:
  - quality_segment: good/bad, split by median per-file SNR proxy from
    compute_audio_snr.py (analysis_results_full_summary/audio_snr.csv) —
    a direct noise-level measurement: 90th-percentile frame RMS (speech
    peak) minus 10th-percentile frame RMS (background/noise floor), both
    in dB, decoded straight from the audio. Higher SNR = good (cleaner
    audio, quieter gaps relative to speech).
  - length_segment: short/long, split by median audio_duration_sec (same
    transcript files).
  - difficulty_segment: clear_violation/ambiguous_violation, GT-POSITIVE
    files only (GT-negative files get "not_applicable" here — there's no
    violation to call clear or ambiguous), split by gt_max_confidence
    (the max GT flag confidence in the file) against --gt-conf-cutoff
    (default 0.7).

Reuses scoring_modes.py (modes) and threshold_analysis.py's existing
_passes_threshold/_signal_value/threshold_metrics_for_group (metric
definitions) exactly as threshold_pp_v6_v8.py does — no new metric
definitions here.

Prompt-version -> scoring: v1 runs use raw only; v6/v7/v8 runs use
raw/sa/pp_full (the same modes threshold_pp_v6_v8.py uses on the dev set).
Partial runs (some v7/v8 full-dataset runs) are included on the files they
finished — n_gt_pos + n_gt_neg on each row gives the file count.

Usage: python3 threshold_by_bucket_full.py
"""
import csv
import glob
import json
import statistics
from pathlib import Path

import config
import analyze_results as ar
from analyze_results import r2, merge_and_write_csv
import threshold_analysis as ta
import scoring_modes as sm
from scoring_modes_pipeline import MATCHER_VERSION, DROP_MODES

OUT_DIR = Path("analysis_results_full_summary")
OUT_DIR.mkdir(exist_ok=True)

TRANSCRIPT_GLOB = "indic_conformer_results/chunk_5s/transcripts/*.json"
SNR_CSV = "analysis_results_full_summary/audio_snr.csv"
GT_CONF_CUTOFF = 0.7

# (folder, prompt_version, [models]) — models=None means "every model dir
# with a per_file/ cache present".
RUNS = [
    ("analysis_results", "v1", None),
    ("analysis_results_promptv6", "v6", None),
    ("analysis_results_promptv7", "v7", None),
    ("analysis_results_promptv8", "v8", None),
]
PROMPT_MODES = {"v1": ["raw"], "v6": ["raw", "sa", "pp_full"],
                "v7": ["raw", "sa", "pp_full"], "v8": ["raw", "sa", "pp_full"]}
NUMERIC_THRESHOLDS = [round(i * 0.1, 2) for i in range(10)] + [0.95, 1.0]
CATEGORIES = list(config.CATEGORY_LABELS.values()) + ["ALL"]


def models_present(folder: str):
    return sorted(p.name for p in Path(folder).iterdir() if p.is_dir() and (p / "per_file").is_dir())


def load_rows(folder: str, model: str):
    per_file_dir = Path(folder) / model / "per_file"
    rows = []
    for fp in sorted(glob.glob(str(per_file_dir / "*.json"))):
        rows.append(json.loads(Path(fp).read_text()))
    return rows


# ------------------------------------------------------- audio_segments ---

def build_audio_segments():
    durs = {}
    for fp in glob.glob(TRANSCRIPT_GLOB):
        d = json.loads(Path(fp).read_text())
        durs[d["file_id"]] = d.get("audio_duration_sec")

    snrs = {}
    with open(SNR_CSV, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get("snr_db")
            snrs[row["file_id"]] = float(v) if v not in (None, "") else None

    dur_vals = [v for v in durs.values() if v is not None]
    snr_vals = [v for v in snrs.values() if v is not None]
    median_dur = statistics.median(dur_vals)
    median_snr = statistics.median(snr_vals)

    # Pull language + GT flags from any one already-scored run (GT is
    # identical across models) — prefer analysis_results (v1, has every
    # file) over the v6 full folder.
    gt_source = "analysis_results/e2b_nothinking/per_file"
    rows_by_fid = {}
    for fp in glob.glob(f"{gt_source}/*.json"):
        d = json.loads(Path(fp).read_text())
        rows_by_fid[d["file_id"]] = d

    segments = {}
    for fid, row in rows_by_fid.items():
        gt_flags = row.get("gt_flags", [])
        gt_positive = len(gt_flags) > 0
        confs = [g.get("confidence") for g in gt_flags if g.get("confidence") is not None]
        gt_max_conf = max(confs) if confs else None

        dur = durs.get(fid)
        snr = snrs.get(fid)
        length_segment = None if dur is None else ("short" if dur < median_dur else "long")
        quality_segment = None if snr is None else ("good" if snr >= median_snr else "bad")

        if not gt_positive:
            difficulty_segment = "not_applicable"
        elif gt_max_conf is None:
            difficulty_segment = "ambiguous_violation"  # fallback: no GT confidence available
        else:
            difficulty_segment = "clear_violation" if gt_max_conf >= GT_CONF_CUTOFF else "ambiguous_violation"

        segments[fid] = {
            "file_id": fid, "language": row.get("language"), "duration_s": dur, "snr_db": snr,
            "gt_positive": gt_positive, "gt_flag_count": len(gt_flags), "gt_max_confidence": gt_max_conf,
            "length_segment": length_segment, "quality_segment": quality_segment,
            "difficulty_segment": difficulty_segment,
        }

    rows_out = list(segments.values())
    ar.write_csv(OUT_DIR / "audio_segments.csv", rows_out,
                 ["file_id", "language", "duration_s", "snr_db", "gt_positive", "gt_flag_count",
                  "gt_max_confidence", "length_segment", "quality_segment", "difficulty_segment"])

    print(f"audio_segments.csv: {len(rows_out)} files. median_duration_s={r2(median_dur)}, median_snr_db={r2(median_snr)}, "
          f"gt_conf_cutoff={GT_CONF_CUTOFF}")
    from collections import Counter
    print("  length:", Counter(r["length_segment"] for r in rows_out))
    print("  quality:", Counter(r["quality_segment"] for r in rows_out))
    print("  difficulty:", Counter(r["difficulty_segment"] for r in rows_out))
    for seg, c in Counter(r["length_segment"] for r in rows_out).items():
        if c < 15:
            print(f"  WARNING: length_segment={seg} has only {c} files (<15)")
    for seg, c in Counter(r["quality_segment"] for r in rows_out).items():
        if c < 15:
            print(f"  WARNING: quality_segment={seg} has only {c} files (<15)")
    for seg, c in Counter(r["difficulty_segment"] for r in rows_out).items():
        if c < 15:
            print(f"  WARNING: difficulty_segment={seg} has only {c} files (<15)")
    return segments


# ------------------------------------------------------------- scoring ----

def ensure_scored(rows, prompt_version, modes):
    for row in rows:
        for flag in row.get("model_flags", []):
            flag["scored"] = {m: sm.adjusted_confidence(flag, m, prompt_version) for m in modes}


def rows_for_mode(rows, mode):
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


def attach_segments(rows, segments, axis):
    """Returns {segment_label: [rows]} for the given axis."""
    out = {}
    for row in rows:
        seg = segments.get(row["file_id"], {}).get(axis)
        if seg is None:
            continue
        out.setdefault(seg, []).append(row)
    return out


ALL_FIELDNAMES = [
    "model", "prompt_version", "scoring_mode", "matcher_version", "bucket_axis", "bucket", "category", "threshold",
    "n_gt_pos", "n_gt_neg", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy", "f1",
    "flag_tp", "flag_fp", "flag_fn", "flag_redundant", "flag_precision", "flag_recall", "n_flags_at_or_above",
]


def redundant_available(rows, category):
    for r in rows:
        for f in r["model_flags"]:
            if category != "ALL" and f.get("category") != category:
                continue
            if f.get("fp_classification"):
                return True
    return False


def n_flags_at_or_above(rows, category, threshold):
    n = 0
    for r in rows:
        for f in r["model_flags"]:
            if category != "ALL" and f.get("category") != category:
                continue
            if ta._passes_threshold(f, threshold, "_scored_signal"):
                n += 1
    return n


def build_rows_for_run(rows, model, prompt_version, segments):
    out = []
    modes = PROMPT_MODES[prompt_version]
    ensure_scored(rows, prompt_version, modes)

    axes = {"overall": {"overall": rows}}
    for axis in ("length_segment", "quality_segment", "difficulty_segment"):
        axes[axis] = attach_segments(rows, segments, axis)

    for mode in modes:
        for axis, bucket_map in axes.items():
            for bucket, bucket_rows in bucket_map.items():
                if not bucket_rows:
                    continue
                mode_rows = rows_for_mode(bucket_rows, mode)
                table = ta.threshold_metrics_table(mode_rows, NUMERIC_THRESHOLDS, signal_key="_scored_signal")
                for category in CATEGORIES:
                    r_avail = redundant_available(bucket_rows, category)
                    for row in table:
                        if row["category"] != category:
                            continue
                        thr_val = None if row["threshold"] == "none" else row["threshold"]
                        n_at = n_flags_at_or_above(mode_rows, category, thr_val)
                        out_row = dict(row)
                        out_row.update({
                            "model": model, "prompt_version": prompt_version, "scoring_mode": mode,
                            "matcher_version": MATCHER_VERSION, "bucket_axis": axis, "bucket": bucket,
                            "n_flags_at_or_above": n_at,
                        })
                        if not r_avail:
                            out_row["flag_redundant"] = ""
                        out.append({k: out_row.get(k, "") for k in ALL_FIELDNAMES})
    return out


def main():
    segments = build_audio_segments()

    all_rows = []
    for folder, prompt_version, model_filter in RUNS:
        if not Path(folder).is_dir():
            print(f"skip {folder}: not found")
            continue
        models = model_filter or models_present(folder)
        print(f"\n[{prompt_version}] {folder}: {models}")
        for model in models:
            rows = load_rows(folder, model)
            if not rows:
                continue
            all_rows.extend(build_rows_for_run(rows, model, prompt_version, segments))
            print(f"  {model}: {len(rows)} files")

    all_rows = [{k: r.get(k, "") for k in ALL_FIELDNAMES} for r in all_rows]
    merge_and_write_csv(OUT_DIR / "threshold_metrics_by_bucket_full.csv", all_rows, ALL_FIELDNAMES,
                         list({r["model"] for r in all_rows}))
    print(f"\nwrote {len(all_rows)} rows to threshold_metrics_by_bucket_full.csv")

    # No-threshold view: every flag counts (threshold "none"), category ALL —
    # one row per model/prompt/mode/bucket, the same rows as in the full table.
    no_threshold = [r for r in all_rows if r["threshold"] == "none" and r["category"] == "ALL"]
    ar.write_csv(OUT_DIR / "bucket_metrics_no_threshold.csv", no_threshold, ALL_FIELDNAMES)
    print(f"wrote {len(no_threshold)} rows to bucket_metrics_no_threshold.csv")

    # overview markdown: category ALL, primary mode per prompt version
    PRIMARY_MODE = {"v1": "raw", "v6": "sa", "v7": "sa", "v8": "sa"}
    md = ["# Threshold-by-bucket overview — full dataset (category ALL, primary scoring mode per run)", ""]
    model_prompt_pairs = sorted({(r["model"], r["prompt_version"]) for r in all_rows})
    buckets_order = [("overall", "overall"), ("length_segment", "short"), ("length_segment", "long"),
                      ("quality_segment", "good"), ("quality_segment", "bad"),
                      ("difficulty_segment", "clear_violation"), ("difficulty_segment", "ambiguous_violation")]
    thresholds_md = [0.0, 0.3, 0.6, 0.8, 0.9, 1.0]
    for model, pv in model_prompt_pairs:
        model_rows = [r for r in all_rows if r["model"] == model and r["prompt_version"] == pv and r["category"] == "ALL"]
        if not model_rows:
            continue
        mode = PRIMARY_MODE.get(pv, "raw")
        md.append(f"## {model} (prompt {pv}, mode={mode})")
        header = "| bucket | " + " | ".join(f"t={t}" for t in thresholds_md) + " |"
        md.append(header)
        md.append("|---" * (len(thresholds_md) + 1) + "|")
        for axis, bucket in buckets_order:
            cands = [r for r in model_rows if r["bucket_axis"] == axis and r["bucket"] == bucket
                     and r["scoring_mode"] == mode]
            if not cands:
                continue
            cells = []
            for t in thresholds_md:
                match = next((r for r in cands if r["threshold"] == t), None)
                if not match:
                    cells.append("")
                    continue
                rec = match["recall"]
                spec = match["specificity"]
                if rec != "" and spec != "":
                    cells.append(f"{rec}/{spec}")
                elif rec != "":
                    cells.append(f"R={rec}")
                elif spec != "":
                    cells.append(f"Spec={spec}")
                else:
                    cells.append("")
            md.append(f"| {bucket} | " + " | ".join(cells) + " |")
        md.append("")
    (OUT_DIR / "threshold_by_bucket_overview.md").write_text("\n".join(md))
    print(f"wrote threshold_by_bucket_overview.md")

    # checks
    print("\n=== Checks ===")
    for folder, prompt_version, model_filter in RUNS:
        if not Path(folder).is_dir():
            continue
        models = model_filter or models_present(folder)
        for model in models:
            rows = load_rows(folder, model)
            if not rows:
                continue
            modes = PROMPT_MODES[prompt_version]
            ensure_scored(rows, prompt_version, modes)
            mode_rows = rows_for_mode(rows, "raw")
            expected_file = ta.aggregate_file_level(mode_rows)
            none_row = next((r for r in all_rows if r["model"] == model and r["prompt_version"] == prompt_version
                              and r["scoring_mode"] == "raw" and r["bucket_axis"] == "overall"
                              and r["category"] == "ALL" and r["threshold"] == "none"), None)
            if none_row is None:
                continue
            for k in ("tp", "fp", "fn", "tn"):
                assert none_row[k] == expected_file[k], f"{prompt_version}/{model}: overall/raw/none {k} mismatch"
            print(f"  {prompt_version}/{model}: OK (raw/none/overall/ALL matches confusion table)")
    print(f"\nAll checks passed. matcher_version={MATCHER_VERSION} used throughout for v6 runs "
          "(v1 runs predate the global matcher's own versioning but share the same codebase).")


if __name__ == "__main__":
    main()
