"""Held-out leaderboard: every full-dataset run, scored on the HELD-OUT files
only (full dataset minus manifest/dev_set_v1.csv), at two fixed settings
(raw >= 0.8 and pp_full >= 0.8) plus each Gemma audio model's picked
setting (mode + threshold, chosen on the dev set).

Read-only over the per-file caches; reuses leaderboard.full_dataset_runs and
threshold_analysis.threshold_metrics_for_group, so numbers follow the same
rules as every other table. Also checks that every run saw the same GT flags
for the files it shares with the others (GT consistency).

Usage: python3 heldout_leaderboard.py
Writes analysis_results_full_summary/heldout_leaderboard.csv
"""
import csv
import hashlib
import json

import config
import scoring_modes as sm
from leaderboard import full_dataset_runs, load_rows, model_input
from threshold_analysis import threshold_metrics_for_group

DEV_IDS = {r["file_id"] for r in csv.DictReader(open(config.PROJECT_ROOT / "manifest" / "dev_set_v1.csv"))}
N_HELDOUT = 331
SIGNAL = "_ho_conf"
DROP_MODES = {"sa_drop_invalid", "pp_full"}

# Picked setting per model, chosen on the dev set (see the Prompt iterations page).
PICKED = {
    ("e2b_nothinking", "v6"): ("pp_full", 0.9),
    ("e2b_thinking", "v8"): ("sa", 0.6),
    ("e4b_nothinking", "v7"): ("pp_full", 0.6),
    ("e4b_thinking", "v7"): ("raw", 0.8),
}


def with_mode(rows, mode, prompt):
    out = []
    for row in rows:
        flags = []
        for f in row.get("model_flags", []):
            conf = f.get("model_confidence") if mode == "raw" else sm.adjusted_confidence(f, mode, prompt)
            if conf is None and mode in DROP_MODES:
                continue
            nf = dict(f)
            nf[SIGNAL] = conf
            flags.append(nf)
        nr = dict(row)
        nr["model_flags"] = flags
        out.append(nr)
    return out


def metrics(rows, mode, threshold, prompt):
    m = threshold_metrics_for_group(with_mode(rows, mode, prompt), "ALL", threshold, SIGNAL)
    return m["recall"], m["specificity"]


def gt_hash(row):
    key = sorted((g.get("category"), g.get("timestamp"), g.get("translation")) for g in row.get("gt_flags", []))
    return hashlib.md5(json.dumps(key).encode()).hexdigest()


def main():
    out = []
    gt_seen = {}
    gt_conflicts = set()
    for prompt, folder in full_dataset_runs():
        for model_dir in sorted(p.parent for p in folder.glob("*/per_file") if p.is_dir()):
            model = model_dir.name
            rows = [r for r in load_rows(folder, model) if r["file_id"] not in DEV_IDS]
            if not rows:
                continue
            for r in rows:
                h = gt_hash(r)
                if gt_seen.setdefault(r["file_id"], h) != h:
                    gt_conflicts.add(r["file_id"])
            rec_raw, spec_raw = metrics(rows, "raw", 0.8, prompt)
            if prompt == "v1":
                rec_pp = spec_pp = None  # v1 has none of the fields pp_full uses
            else:
                rec_pp, spec_pp = metrics(rows, "pp_full", 0.8, prompt)
            picked = PICKED.get((model, prompt))
            rec_pk = spec_pk = None
            if picked:
                rec_pk, spec_pk = metrics(rows, picked[0], picked[1], prompt)
            n_pos = sum(1 for r in rows if r.get("gt_flags"))
            out.append({
                "model": model, "input": model_input(model), "prompt": prompt,
                "heldout_files": len(rows), "complete": len(rows) >= N_HELDOUT - 2,
                "gt_pos_files": n_pos, "gt_neg_files": len(rows) - n_pos,
                "raw08_recall": rec_raw, "raw08_spec": spec_raw,
                "pp08_recall": rec_pp, "pp08_spec": spec_pp,
                "picked": f"{picked[0]} @ {picked[1]}" if picked else "",
                "picked_recall": rec_pk, "picked_spec": spec_pk,
            })
    out.sort(key=lambda r: (not r["complete"], -((r["raw08_recall"] or 0) + (r["raw08_spec"] or 0))))
    path = config.PROJECT_ROOT / "analysis_results_full_summary" / "heldout_leaderboard.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
    print(f"GT consistency: {len(gt_seen)} held-out files seen, {len(gt_conflicts)} with differing GT flags across runs")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
