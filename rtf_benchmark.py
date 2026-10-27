"""Clean, local, end-to-end timing of the transcript-based moderation pipeline
on a fixed set of calls: Indic-Transcribe-Core transcription (10 s chunks)
followed by Gemma moderation with prompt v8 (E2B/E4B, thinking or not).

Each stage runs alone on this machine (one model loaded at a time), with the
same code the real runs use (indic_transcribe_core_pipeline.transcribe_file,
gemma_local_text.classify_text + gemma_local.parse_and_score_flags). Model
load time is recorded separately and excluded from per-call RTF.

Output in rtf_benchmark/: asr_<file_id>.json transcripts, timing.csv
(one row per call per stage), and summary.csv (RTF per stage and pipeline).

Usage:
  python3 rtf_benchmark.py asr --files a,b,...
  python3 rtf_benchmark.py moderate --model e2b [--thinking] --files a,b,...
  python3 rtf_benchmark.py summary
"""
import argparse
import csv
import json
import statistics as st
import time
from pathlib import Path

import config
import dataset_v2 as dsv2

OUT = config.PROJECT_ROOT / "rtf_benchmark"
TIMING = OUT / "timing.csv"
COLS = ["stage", "file_id", "language", "audio_duration_sec", "seconds", "rtf", "detail", "model_load_seconds", "device"]


def log_row(row):
    new = not TIMING.exists()
    with TIMING.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def records(files):
    recs = {r.file_id: r for r in dsv2.load_dataset_v2(config.DOSTT_DIR)}
    return [recs[f] for f in files.split(",")]


def asr(args):
    import indic_transcribe_core_pipeline as itc
    from pipeline_logging import StageLogger
    logger = StageLogger("rtf_benchmark_asr")
    t0 = time.time()
    model = itc.load_model()
    load_s = round(time.time() - t0, 1)
    device = str(getattr(model, "device", "")) or "default"
    print(f"Indic-Transcribe-Core loaded in {load_s}s ({device})", flush=True)
    for rec in records(args.files):
        t0 = time.perf_counter()
        wav = itc.load_audio(rec.path)
        segs, _full, _ = itc.transcribe_file(model, wav, itc.LANGUAGE_CODES[rec.language], 10.0, logger, rec.file_id)
        secs = time.perf_counter() - t0
        dur = len(wav) / itc.SAMPLE_RATE
        (OUT / f"asr_{rec.file_id}.json").write_text(json.dumps({"file_id": rec.file_id, "segments": segs}, ensure_ascii=False))
        log_row({"stage": "asr_itc10s", "file_id": rec.file_id, "language": rec.language, "audio_duration_sec": round(dur, 1),
                 "seconds": round(secs, 2), "rtf": round(secs / dur, 4), "detail": f"{len(segs)} segments",
                 "model_load_seconds": load_s, "device": device})
        print(f"asr {rec.file_id} {round(dur)}s audio: {secs:.1f}s (RTF {secs / dur:.3f})", flush=True)


def moderate(args):
    import gemma_local_text as g
    import prompt_loader
    from gemma_local import load_model, parse_and_score_flags
    from pipeline_logging import StageLogger
    g.PROMPT_PATH = config.PROJECT_ROOT / "prompt_v8.py"
    g.RAW_SCHEMA_STR = json.dumps(prompt_loader.schema_module_for_prompt(g.PROMPT_PATH).raw_model_output_json_schema())
    g.THINKING = args.thinking
    stage = f"moderate_{args.model}{'_thinking' if args.thinking else ''}"
    logger = StageLogger(f"rtf_benchmark_{stage}")
    t0 = time.time()
    model, processor = load_model(args.model)
    load_s = round(time.time() - t0, 1)
    print(f"{stage}: model loaded in {load_s}s on {model.device}", flush=True)
    for rec in records(args.files):
        segs = json.loads((OUT / f"asr_{rec.file_id}.json").read_text())["segments"]
        text = "\n".join(f"[{s['t']}] {s['text']}" for s in segs if s["text"])
        dur = next(float(r["audio_duration_sec"]) for r in csv.DictReader(TIMING.open())
                   if r["stage"] == "asr_itc10s" and r["file_id"] == rec.file_id)
        t0 = time.perf_counter()
        answer, _raw, token_infos = g.classify_text(model, processor, text, logger)
        flags, _ = parse_and_score_flags(answer, token_infos, chunk_offset_sec=0.0, logger=logger)
        secs = time.perf_counter() - t0
        g.free_device_memory()
        log_row({"stage": stage, "file_id": rec.file_id, "language": rec.language, "audio_duration_sec": dur,
                 "seconds": round(secs, 2), "rtf": round(secs / dur, 4), "detail": f"{len(flags)} flags",
                 "model_load_seconds": load_s, "device": str(model.device)})
        print(f"{stage} {rec.file_id} {round(dur)}s audio: {secs:.1f}s (RTF {secs / dur:.3f}), {len(flags)} flags", flush=True)


def audio_moderate(args):
    """Direct audio moderation: Gemma hears 28 s chunks with prompt v8, through
    gemma_local.process_file exactly as the real audio runs do."""
    import tempfile
    import gemma_local as gl
    import prompt_loader
    from pipeline_logging import StageLogger
    gl.PROMPT_PATH = config.PROJECT_ROOT / "prompt_v8.py"
    gl.RAW_SCHEMA_STR = json.dumps(prompt_loader.schema_module_for_prompt(gl.PROMPT_PATH).raw_model_output_json_schema())
    stage = f"audio_moderate_{args.model}{'_thinking' if args.thinking else ''}"
    logger = StageLogger(f"rtf_benchmark_{stage}")
    t0 = time.time()
    model, processor = gl.load_model(args.model)
    load_s = round(time.time() - t0, 1)
    print(f"{stage}: model loaded in {load_s}s on {model.device}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="rtf_audio_"))
    from mutagen import File as MutagenFile
    for rec in records(args.files):
        info = MutagenFile(rec.path)
        dur = float(info.info.length)
        t0 = time.perf_counter()
        result, _raw = gl.process_file(model, processor, rec, args.model, args.thinking, gl.DEFAULT_CHUNK_SECONDS, logger, tmp)
        secs = time.perf_counter() - t0
        import gc, torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log_row({"stage": stage, "file_id": rec.file_id, "language": rec.language, "audio_duration_sec": round(dur, 1),
                 "seconds": round(secs, 2), "rtf": round(secs / dur, 4),
                 "detail": f"{len(result.flags)} flags, {result.chunks_total} chunks, {result.chunks_failed} failed",
                 "model_load_seconds": load_s, "device": str(model.device)})
        print(f"{stage} {rec.file_id} {round(dur)}s audio: {secs:.1f}s (RTF {secs / dur:.3f}), {len(result.flags)} flags", flush=True)


def summary(_args):
    rows = list(csv.DictReader(TIMING.open()))
    by = {}
    for r in rows:
        by.setdefault(r["stage"], {})[r["file_id"]] = r
    asr_rows = by.get("asr_itc10s", {})
    out = []
    for stage, d in by.items():
        rtf = [float(r["rtf"]) for r in d.values()]
        total = sum(float(r["seconds"]) for r in d.values()) / sum(float(r["audio_duration_sec"]) for r in d.values())
        rec = {"stage": stage, "calls": len(d), "rtf_median": round(st.median(rtf), 3), "rtf_mean": round(st.mean(rtf), 3),
               "rtf_min": round(min(rtf), 3), "rtf_max": round(max(rtf), 3), "rtf_overall": round(total, 3)}
        if stage.startswith("moderate_") and asr_rows:
            both = [f for f in d if f in asr_rows]
            secs = sum(float(d[f]["seconds"]) + float(asr_rows[f]["seconds"]) for f in both)
            dur = sum(float(asr_rows[f]["audio_duration_sec"]) for f in both)
            per = [(float(d[f]["seconds"]) + float(asr_rows[f]["seconds"])) / float(asr_rows[f]["audio_duration_sec"]) for f in both]
            rec.update({"pipeline_rtf_median": round(st.median(per), 3), "pipeline_rtf_overall": round(secs / dur, 3)})
        out.append(rec)
    with (OUT / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in out for k in r}, key=lambda k: list(out[-1]).index(k) if k in out[-1] else 99))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["asr", "moderate", "audio_moderate", "summary"])
    ap.add_argument("--files", default="")
    ap.add_argument("--model", choices=["e2b", "e4b"])
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"asr": asr, "moderate": moderate, "audio_moderate": audio_moderate, "summary": summary}[a.cmd](a)
