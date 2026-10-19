"""Laya zero-shot verifier experiment (Mode B), dev set.

For each candidate flag from a model's run, Laya answers yes/no questions
about the flag (laya_common.build_questions) given a state built from the
quote and, optionally, a transcript window around the flag's timestamp. Every
raw Laya output is saved (laya_common.save_raw), per-file timing is logged,
and the scores are evaluated with the same metric code as the existing
threshold tables.

Candidates: flags whose pp_full confidence (scoring_modes) is >= 0.3. Flags
below that are never sent to Laya; in the file-level metrics they keep a
score of 0 for every Laya-based score, so all thresholds >= 0.3 are exact.

Scores per candidate:
  gemma_only     pp_full confidence (baseline; must reproduce the existing
                 pp_full threshold numbers for that run)
  laya_cat       Laya P(yes) for the flag's own category question
  laya_combined  laya_cat * (1 - about_not_doing) * quote_supported
  gemma_x_laya   gemma_only * laya_combined
  gemma_x_cat    gemma_only * laya_cat

Usage:
  python3 laya_experiment.py run      # Laya requests -> raw JSONL + timing
  python3 laya_experiment.py score    # metrics from the saved raw JSONL
"""
import csv
import json
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import laya_common as L
import scoring_modes as sm

# (candidate model key, prompt version, per_file dir). Gemini 3.1 has no
# separate dev-set run; its full-dataset v6 run covers all 50 dev calls.
RUNS = [
    ("e2b_nothinking", "v6", "analysis_results_dev_promptv6/e2b_nothinking/per_file"),
    ("gemini-3.1-flash-lite", "v6", "analysis_results_promptv6/gemini-3.1-flash-lite/per_file"),
]
SOURCES = ["conformer", "gt"]
DEFS = ["short"]
CTXS = [("quote_only", None), ("pm10", 10), ("pm30", 30)]
MAX_LEN, HEAD_MAX_LEN = 2048, 1024
MIN_CONF = 0.3
RUN_NAME = "step2"
THRESHOLDS = [0.0, 0.01, 0.05] + [round(i / 10, 1) for i in range(1, 10)] + [0.95, 0.99, None]
CAT2Q = {v: k for k, v in L.CATEGORIES.items()}
SCORES = ["gemma_only", "laya_cat", "laya_combined", "gemma_x_laya", "gemma_x_cat"]


def dev_rows(per_file_dir):
    dev = {f["file_id"] for f in L.dev_files()}
    rows = [json.loads(p.read_text()) for p in sorted(Path(per_file_dir).glob("*.json")) if p.stem in dev]
    return rows


def candidates(row, prompt):
    for i, f in enumerate(row["model_flags"]):
        c = sm.adjusted_confidence(f, "pp_full", prompt)
        if c is None or c < MIN_CONF or f.get("category") not in CAT2Q:
            continue
        yield i, f, c


def other_load():
    """Other heavy local processes sharing the machine, recorded with the timings."""
    try:
        out = subprocess.run(["pgrep", "-fl", "gemma_local|gemma_local_text|openrouter_text"],
                             capture_output=True, text=True).stdout
        return [l.split(" ", 1)[1][:60] for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def run():
    r = L.router()
    timing_path = L.OUT_ROOT / "laya_timing_per_file.csv"
    new = not timing_path.exists()
    tfh = timing_path.open("a", newline="")
    tw = csv.writer(tfh)
    if new:
        tw.writerow(["candidate_model", "prompt_version", "source", "defs", "ctx", "file_id", "n_candidates",
                     "laya_ms_total", "laya_ms_per_candidate", "wall_ms", "device", "concurrent_jobs", "host"])
    device = str(r.load(L.MODEL).device)
    load = ";".join(other_load())
    for model, prompt, pdir in RUNS:
        rows = dev_rows(pdir)
        for src in SOURCES:
            for defs in DEFS:
                for ctx, half in CTXS:
                    if src == "gt" and ctx == "quote_only":
                        continue  # identical state to conformer/quote_only
                    t_cfg = time.time()
                    n_req = 0
                    for row in rows:
                        fid = row["file_id"]
                        segs = L.load_segments(src, fid) if half else None
                        wall0 = time.time()
                        laya_ms = 0.0
                        n = 0
                        for idx, f, gc in candidates(row, prompt):
                            t = L.mmss_to_sec(f.get("timestamp"))
                            state = {"quote_native": f.get("excerpt", ""), "quote_english": f.get("translation", "")}
                            if half:
                                state["transcript_window"] = L.window_text(segs, t or 0, half)
                            q = L.build_questions(defs, "B", f.get("translation", ""))
                            t0 = time.perf_counter()
                            res = r.predict(state, q, model=L.MODEL, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
                            ms = (time.perf_counter() - t0) * 1000
                            laya_ms += ms
                            n += 1
                            L.save_raw("B", src, defs, ctx, {
                                "candidate_model": model, "candidate_prompt_version": prompt,
                                "file_id": fid, "flag_index": idx,
                                "candidate": {k: f.get(k) for k in ("category", "timestamp", "excerpt", "translation",
                                                                     "model_confidence", "matched", "matched_gt_category")},
                                "gemma_pp_full": gc, "ts_unreliable": t in (None, 0),
                                "request": {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN, "model": L.MODEL,
                                            "latency_ms": round(ms, 1)},
                                "state": state, "question_ids": list(q), "laya": res,
                            }, run_name=f"{RUN_NAME}_{model}")
                        tw.writerow([model, prompt, src, defs, ctx, fid, n, round(laya_ms, 1),
                                     round(laya_ms / n, 1) if n else "", round((time.time() - wall0) * 1000, 1),
                                     device, load, socket.gethostname()])
                        tfh.flush()
                        n_req += n
                    print(f"{model} {src} {defs} {ctx}: {n_req} requests in {time.time() - t_cfg:.0f}s", flush=True)
    tfh.close()


# ---------------------------------------------------------------- scoring

def load_raw(model, src, defs, ctx):
    p = L.OUT_ROOT / "mode_B" / src / defs / ctx / f"{RUN_NAME}_{model}.jsonl"
    out = {}
    for line in p.open():
        d = json.loads(line)
        a = d["laya"]["answers"]
        cat = CAT2Q[d["candidate"]["category"]]
        lc = a[cat]["noul"]
        comb = lc * (1 - a["about_not_doing"]["noul"]) * a["quote_supported"]["noul"]
        out[(d["file_id"], d["flag_index"])] = {"laya_cat": lc, "laya_combined": comb,
                                                "about_not_doing": a["about_not_doing"]["noul"],
                                                "truncated": d["laya"]["usage"].get("truncated")}
    return out


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return round(s / (len(pos) * len(neg)), 3)


def auprc(scores_labels):
    pairs = sorted(scores_labels, key=lambda x: -x[0])
    tp = 0
    total_pos = sum(l for _, l in pairs)
    if not total_pos:
        return None
    ap = 0.0
    for i, (_, l) in enumerate(pairs, 1):
        if l:
            tp += 1
            ap += tp / i
    return round(ap / total_pos, 3)


def score():
    from threshold_analysis import threshold_metrics_for_group  # pulls in analyze_results; not needed by run()
    metric_rows, flag_rows = [], []
    for model, prompt, pdir in RUNS:
        rows = dev_rows(pdir)
        for src in SOURCES:
            for defs in DEFS:
                for ctx, _ in CTXS:
                    raw_src = "conformer" if (src == "gt" and ctx == "quote_only") else src
                    try:
                        laya = load_raw(model, raw_src, defs, ctx)
                    except FileNotFoundError:
                        continue
                    scored = []
                    for row in rows:
                        flags = []
                        for i, f in enumerate(row["model_flags"]):
                            g = sm.adjusted_confidence(f, "pp_full", prompt)
                            if g is None:
                                continue  # dropped by pp_full
                            l = laya.get((row["file_id"], i))
                            nf = dict(f)
                            nf["gemma_only"] = g
                            if l is None:  # never sent to Laya (pp_full < 0.3)
                                nf.update({"laya_cat": 0.0, "laya_combined": 0.0, "gemma_x_laya": 0.0, "gemma_x_cat": 0.0})
                            else:
                                nf.update({"laya_cat": l["laya_cat"], "laya_combined": l["laya_combined"],
                                           "gemma_x_laya": g * l["laya_combined"], "gemma_x_cat": g * l["laya_cat"]})
                            nf["_cand"] = l is not None
                            flags.append(nf)
                        nr = dict(row)
                        nr["model_flags"] = flags
                        scored.append(nr)
                    cands = [f for r_ in scored for f in r_["model_flags"] if f["_cand"]]
                    n_trunc = sum(1 for v in laya.values() if v["truncated"])
                    for sname in SCORES:
                        pos = [f[sname] for f in cands if f.get("matched")]
                        neg = [f[sname] for f in cands if not f.get("matched")]
                        flag_rows.append({"candidate_model": model, "prompt_version": prompt, "source": src, "defs": defs,
                                          "ctx": ctx, "score_name": sname, "n_candidates": len(cands),
                                          "n_matched": len(pos), "flag_auroc": auc(pos, neg),
                                          "flag_auprc": auprc([(f[sname], bool(f.get("matched"))) for f in cands]),
                                          "truncation_rate": round(n_trunc / len(laya), 3) if laya else None})
                        for cat in ["ALL"] + list(L.CATEGORIES.values()):
                            for t in THRESHOLDS:
                                m = threshold_metrics_for_group(scored, cat, t, sname)
                                metric_rows.append({"candidate_model": model, "prompt_version": prompt, "source": src,
                                                    "defs": defs, "ctx": ctx, "score_name": sname, "language": "ALL",
                                                    "category": cat, "threshold": m["threshold"],
                                                    **{k: m[k] for k in ("tp", "fp", "fn", "tn", "recall", "precision",
                                                                         "specificity", "accuracy", "f1")}})
    write_csv(L.OUT_ROOT / "laya_mode_b_threshold_metrics.csv", metric_rows)
    write_csv(L.OUT_ROOT / "laya_mode_b_flag_auroc.csv", flag_rows)
    return metric_rows, flag_rows


def write_csv(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
