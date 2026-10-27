"""RTF (processing time / audio duration) per model, averaged over every
existing run in logs/ — no new runs. Mixes hardware (Mac MPS, RunPod GPUs,
sharded and not), as requested: the result is an average across setups.

Per-call time = gap between consecutive "[i/N] <file_id>" lines within one
run (a run starts at a "Loading google/gemma-4-..." line); the last call of a
run counts only if a "Done in" line closes it. Audio duration comes from
analysis_results_full_summary/audio_segments.csv.

Stall outliers removed:
  1. calls whose log has a memory-pressure warning ("chunk ... took Ns") with N >= 300 s
  2. per model group, RTF > Q3 + 3 * IQR

Also includes Gemma 31B (OpenRouter, inference_times.csv) and the Gemini runs
that recorded per-call time (inference_time_sec in their results).

Output: analysis_results_full_summary/rtf_by_model.csv (+ rtf_per_call.csv).
"""
import csv
import glob
import json
import re
import statistics as st
from datetime import datetime
from pathlib import Path

import config

ROOT = config.PROJECT_ROOT
OUT = ROOT / "analysis_results_full_summary"
TS = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\] \[(\w+)\] (.*)$")
FILE_LINE = re.compile(r"\[(\d+)/(\d+)\] (rec_\S+?\.mp3)")
LOAD = re.compile(r"Loading google/gemma-4-([A-Za-z0-9]+)-it \(([^)]*)\)")
STALL = re.compile(r"(rec_\S+?\.mp3) chunk@\S+ took (\d+)s")
STALL_SEC = 300


def durations():
    with (OUT / "audio_segments.csv").open() as fh:
        return {r["file_id"]: float(r["duration_s"]) for r in csv.DictReader(fh) if r["duration_s"]}


def parse_gemma_log(path: Path, dur):
    """-> list of per-call dicts."""
    calls, run, open_call = [], None, None
    text_log = "text" in path.name

    def close(at):
        nonlocal open_call
        if open_call and run:
            secs = (at - open_call["start"]).total_seconds()
            fid = open_call["file_id"]
            if fid in dur and secs > 0:
                calls.append({"log": path.name, "model": run["model"], "thinking": run["thinking"],
                              "input": run["input"], "file_id": fid, "seconds": secs,
                              "audio_sec": dur[fid], "rtf": secs / dur[fid], "stall": open_call["stall"]})
        open_call = None

    for line in path.read_text(errors="ignore").splitlines():
        m = TS.match(line)
        if not m:
            continue
        t = datetime.fromisoformat(m.group(1))
        msg = m.group(3)
        lm = LOAD.search(msg)
        if lm:
            open_call = None  # a restart: the previous open call has no end point
            mode = lm.group(2)
            # audio logs say "thinking=True|False"; text logs say "text-only, thinking|non-thinking"
            thinking = ("thinking=True" in mode) if "thinking=" in mode else ("non-thinking" not in mode)
            # older text logs (e.g. e2b_text_nothinking_v5_local) predate --transcript-source: Conformer
            src = "itc" if "itc" in path.name else "conformer"
            run = {"model": lm.group(1).lower(), "thinking": thinking,
                   "input": f"transcript ({src})" if (text_log or "text-only" in mode) else "audio"}
            continue
        fm = FILE_LINE.search(msg)
        if fm and run:
            close(t)
            open_call = {"file_id": fm.group(3), "start": t, "stall": False}
            continue
        sm = STALL.search(msg)
        if sm and open_call and int(sm.group(2)) >= STALL_SEC:
            open_call["stall"] = True
        if msg.startswith("Done in"):
            close(t)
    return calls


def summarize(calls, key):
    groups = {}
    for c in calls:
        groups.setdefault(key(c), []).append(c)
    rows = []
    for k, cs in sorted(groups.items()):
        clean = [c for c in cs if not c["stall"]]
        r = sorted(c["rtf"] for c in clean)
        if len(r) >= 4:
            q1, q3 = r[len(r) // 4], r[(3 * len(r)) // 4]
            cut = q3 + 3 * (q3 - q1)
            kept = [c for c in clean if c["rtf"] <= cut]
        else:
            kept = clean
        if not kept:
            continue
        rtf = [c["rtf"] for c in kept]
        rows.append({"model": k[0], "mode": k[1], "input": k[2], "calls_logged": len(cs),
                     "removed_stall_warning": len(cs) - len(clean), "removed_outlier": len(clean) - len(kept),
                     "calls_used": len(kept), "rtf_mean": round(st.mean(rtf), 3), "rtf_median": round(st.median(rtf), 3),
                     "rtf_overall": round(sum(c["seconds"] for c in kept) / sum(c["audio_sec"] for c in kept), 3),
                     "logs": len({c["log"] for c in kept})})
    return rows


def main():
    dur = durations()
    calls = []
    for p in sorted(Path(ROOT / "logs").glob("*.log")):
        if p.name.startswith(("rtf_benchmark", "gemini_", "openrouter_", "semantic", "analyze")):
            continue
        calls += parse_gemma_log(p, dur)
    # Gemma 31B via OpenRouter: measured request time per call
    for mode in ("nothinking", "thinking"):
        f = ROOT / "gemma_results_promptv8" / f"31b_itc10s_text_{mode}" / "inference_times.csv"
        for r in csv.DictReader(f.open()):
            if r["inference_time_sec"] and r["file_id"] in dur:
                s = float(r["inference_time_sec"])
                calls.append({"log": f.name, "model": "31b", "thinking": mode == "thinking", "input": "transcript (itc, OpenRouter)",
                              "file_id": r["file_id"], "seconds": s, "audio_sec": dur[r["file_id"]],
                              "rtf": s / dur[r["file_id"]], "stall": False})
    # Gemini runs that recorded per-call time
    for d in glob.glob(str(ROOT / "gemini_results*" / "*" / "results")):
        name = Path(d).parent.name
        for p in Path(d).glob("*.json"):
            j = json.loads(p.read_text())
            if j.get("inference_time_sec") and j["file_id"] in dur:
                inp = (j.get("inference_meta") or {}).get("input", "?")
                calls.append({"log": name, "model": name, "thinking": False, "input": f"{inp} (API)",
                              "file_id": j["file_id"], "seconds": j["inference_time_sec"], "audio_sec": dur[j["file_id"]],
                              "rtf": j["inference_time_sec"] / dur[j["file_id"]], "stall": False})
    rows = summarize(calls, lambda c: (c["model"], "thinking" if c["thinking"] else "non-thinking", c["input"]))
    for path, data in ((OUT / "rtf_by_model.csv", rows), (OUT / "rtf_per_call.csv", calls)):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print(f"{'model':38} {'mode':13} {'input':30} {'logged':>6} {'stall':>5} {'outl':>4} {'used':>5} {'mean':>6} {'median':>6} {'overall':>7}")
    for r in rows:
        print(f"{r['model']:38} {r['mode']:13} {r['input']:30} {r['calls_logged']:6} {r['removed_stall_warning']:5} "
              f"{r['removed_outlier']:4} {r['calls_used']:5} {r['rtf_mean']:6} {r['rtf_median']:6} {r['rtf_overall']:7}")


if __name__ == "__main__":
    main()
