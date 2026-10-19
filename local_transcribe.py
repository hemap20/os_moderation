"""Transcribe call audio with a LOCAL Gemma model (E2B/E4B, thinking or not),
timing every call, and score it against the Gemini ground-truth transcript.

Gemma accepts <=30 s of audio per request, so each call is split into 28 s
chunks (gemma_local.chunk_audio, same as the moderation runs); each chunk is
transcribed verbatim in the native script with chunk-relative MM:SS
timestamps, which are shifted back to call time.

Output: transcripts_local/gemma_<run>/<file_id>.json and
transcripts_local/local_transcription_timing.csv (per call: time, chunks,
tokens, WER/CER vs the ground-truth transcript, parse failures).

Usage: python3 local_transcribe.py --model e2b [--thinking] --files id1,id2,...
"""
import argparse
import csv
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import asr_metrics
import config
import dataset_v2 as dsv2
from gemma_local import DEFAULT_CHUNK_SECONDS, MODEL_IDS, chunk_audio, load_model

OUT_ROOT = config.PROJECT_ROOT / "transcripts_local"
TIMING_CSV = OUT_ROOT / "local_transcription_timing.csv"

PROMPT = """
You are a precise audio transcription engine. Transcribe the ENTIRE attached audio clip from start to end,
in the language(s) actually spoken (do not translate). The clip is part of a phone/video call in {language},
often mixed with English words.

Rules:
- Cover the full clip. Do not stop early or summarize.
- Break the transcript into short lines, each with a timestamp "t" in MM:SS marking when the line starts,
  relative to the start of THIS clip.
- Transcribe verbatim in the NATIVE SCRIPT of {language}. Do NOT romanize; write English words embedded in
  the sentence in the native script too.
- If a line is inaudible, write "[inaudible]" for it rather than guessing. If the clip is silent, return
  an empty list.

Return ONLY a JSON object of this form:
{{"segments": [{{"t": "MM:SS", "text": "..."}}]}}
""".strip()

THINKING = False


def mmss(t):
    m = re.fullmatch(r"\s*(\d+):(\d{1,2})\s*", t or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def fmt(sec):
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def transcribe_chunk(model, processor, wav: Path, language: str):
    import torch
    msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(language=language.capitalize())},
                                         {"type": "audio", "audio": str(wav)}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=True, enable_thinking=THINKING).to(model.device)
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4096 if THINKING else 2048, do_sample=False)
    gen = out[0][n_in:]
    if THINKING:
        raw = processor.decode(gen, skip_special_tokens=False)
        parsed = processor.parse_response(raw, prefix=processor.decode(inputs["input_ids"][0], skip_special_tokens=False))
        text, thinking = parsed.get("content", raw), parsed.get("thinking") or ""
        n_think = len(processor.tokenizer(thinking, add_special_tokens=False)["input_ids"]) if thinking else 0
    else:
        text, n_think = processor.decode(gen, skip_special_tokens=True), 0
    return text, int(n_in), int(gen.shape[-1]), n_think


def parse_segments(text):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        d, _ = json.JSONDecoder().raw_decode(text)
    segs = d.get("segments", []) if isinstance(d, dict) else d
    return [s for s in segs if isinstance(s, dict) and s.get("text")]


def free():
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    global THINKING
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["e2b", "e4b"], required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--files", required=True)
    args = ap.parse_args()
    THINKING = args.thinking
    run = f"{args.model}_thinking" if THINKING else args.model
    out_dir = OUT_ROOT / f"gemma_{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = {r.file_id: r for r in dsv2.load_dataset_v2(config.DOSTT_DIR)}

    t_load = time.time()
    model, processor = load_model(args.model)
    load_s = round(time.time() - t_load, 1)
    print(f"{MODEL_IDS[args.model]} ({'thinking' if THINKING else 'non-thinking'}) loaded in {load_s}s", flush=True)

    new = not TIMING_CSV.exists()
    fh = TIMING_CSV.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["model", "file_id", "language", "audio_duration_sec", "chunks", "chunks_parse_failed", "segments",
                    "input_tokens", "output_tokens", "thinking_tokens", "gen_seconds", "wall_seconds",
                    "sec_per_audio_minute", "wer_vs_gt", "cer_vs_gt", "model_load_seconds"])
    tmp_root = Path(tempfile.mkdtemp(prefix="gemma_asr_"))
    for fid in args.files.split(","):
        rec = recs[fid]
        t0 = time.time()
        chunks = chunk_audio(rec.path, DEFAULT_CHUNK_SECONDS, tmp_root / fid)
        segs, failed, n_in, n_out, n_think, gen_s = [], 0, 0, 0, 0, 0.0
        for wav, offset in chunks:
            g0 = time.perf_counter()
            text, a, b, c = transcribe_chunk(model, processor, wav, rec.language)
            gen_s += time.perf_counter() - g0
            n_in, n_out, n_think = n_in + a, n_out + b, n_think + c
            try:
                for s in parse_segments(text):
                    rel = mmss(s.get("t")) or 0
                    segs.append({"t": fmt(offset + min(rel, DEFAULT_CHUNK_SECONDS)), "text": s["text"]})
            except Exception:  # noqa: BLE001
                failed += 1
            free()
        wall = time.time() - t0
        hyp = " ".join(s["text"] for s in segs)
        gt = json.loads((rec.transcripts_dir / f"{fid}.json").read_text())["transcript"].get("full_text", "")
        wer = asr_metrics.word_error_rate(gt, hyp)["wer"]
        cer = asr_metrics.char_error_rate(gt, hyp)["cer"]
        dur = sum(1 for _ in chunks) and chunks[-1][1] + DEFAULT_CHUNK_SECONDS
        from mutagen import File as MutagenFile
        info = MutagenFile(rec.path)
        dur = float(info.info.length) if info is not None and info.info is not None else dur
        (out_dir / f"{fid}.json").write_text(json.dumps({
            "file_id": fid, "language": rec.language, "model": MODEL_IDS[args.model], "thinking": THINKING,
            "chunk_seconds": DEFAULT_CHUNK_SECONDS, "seconds": round(wall, 1), "audio_duration_sec": dur,
            "wer_vs_gt": wer, "cer_vs_gt": cer, "segments": segs, "full_text": hyp,
        }, ensure_ascii=False, indent=2))
        w.writerow([run, fid, rec.language, round(dur, 1), len(chunks), failed, len(segs), n_in, n_out, n_think,
                    round(gen_s, 1), round(wall, 1), round(wall / (dur / 60), 1) if dur else "",
                    round(wer, 3) if wer is not None else "", round(cer, 3) if cer is not None else "", load_s])
        fh.flush()
        print(f"{run} {fid} {rec.language} {round(dur)}s audio, {len(chunks)} chunks: {round(wall, 1)}s, "
              f"WER={wer and round(wer, 2)} CER={cer and round(cer, 2)}{f' ({failed} chunk parse failures)' if failed else ''}",
              flush=True)
    fh.close()
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
