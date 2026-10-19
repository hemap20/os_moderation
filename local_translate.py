"""Translate ITC transcripts into English with a LOCAL Gemma model (non-thinking)
and time it, for comparison with the Gemini 3.1 Flash Lite translations.

Same prompt, chunking (CHUNK segments per request), repetition collapsing and
alignment checks as translate_transcripts.py; only the model differs.

Output: translations/itc10s_gemma_<model>/<file_id>.json (same shape as the
Gemini translations) and translations/local_translation_timing.csv with one
row per call: model, call length, segments, requests, generation seconds,
output tokens, aligned or not.

Usage: python3 local_translate.py --model e2b --files id1,id2,...
"""
import argparse
import csv
import json
import time
from pathlib import Path

import config
import translate_transcripts as tt
from gemma_local import MODEL_IDS, load_model

OUT_ROOT = config.PROJECT_ROOT / "translations"
TIMING_CSV = OUT_ROOT / "local_translation_timing.csv"


THINKING = False


def generate(model, processor, prompt):
    """Returns (answer_text, input_tokens, output_tokens, thinking_tokens).
    Thinking mode strips the reasoning with processor.parse_response, exactly as
    gemma_local.classify_chunk does, and gets twice the token budget."""
    import torch
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=True, enable_thinking=THINKING).to(model.device)
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8192 if THINKING else 4096, do_sample=False)
    gen = out[0][n_in:]
    if not THINKING:
        return processor.decode(gen, skip_special_tokens=True), n_in, int(gen.shape[-1]), 0
    raw = processor.decode(gen, skip_special_tokens=False)
    prefix = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
    parsed = processor.parse_response(raw, prefix=prefix)
    answer = parsed.get("content", raw)
    thinking = parsed.get("thinking") or ""
    think_tokens = len(processor.tokenizer(thinking, add_special_tokens=False)["input_ids"]) if thinking else 0
    return answer, n_in, int(gen.shape[-1]), think_tokens


def free():
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def translate_chunk(model, processor, language, segs, stats):
    payload = json.dumps([{"t": s["t"], "text": tt.collapse_repeats(s["text"])} for s in segs], ensure_ascii=False)
    prompt = tt.PROMPT.format(language=language.capitalize(), segments_json=payload)
    t0 = time.perf_counter()
    text, n_in, n_out, n_think = generate(model, processor, prompt)
    stats["gen_seconds"] += time.perf_counter() - t0
    stats["requests"] += 1
    stats["input_tokens"] += n_in
    stats["output_tokens"] += n_out
    stats["thinking_tokens"] += n_think
    free()
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        out = json.loads(text).get("segments", [])
    except Exception as exc:
        raise ValueError(f"malformed JSON reply: {exc}")
    if len(out) != len(segs):
        raise ValueError(f"segment count {len(out)} != {len(segs)}")
    if any(a["t"] != b.get("t") for a, b in zip(segs, out)):
        raise ValueError("timestamp mismatch")
    return [o.get("en", "") for o in out]


def translate_call(model, processor, language, segs, stats, size=tt.CHUNK, depth=0):
    english = []
    for i in range(0, len(segs), size):
        part = segs[i:i + size]
        last = None
        for _ in range(2):
            try:
                english += translate_chunk(model, processor, language, part, stats)
                break
            except ValueError as exc:
                last = exc
                stats["retries"] += 1
        else:
            if len(part) == 1 or depth >= 4:
                raise ValueError(f"could not align chunk at segment {i}: {last}")
            english += translate_call(model, processor, language, part, stats, max(1, len(part) // 2), depth + 1)
    return english


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["e2b", "e4b"], required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()
    global THINKING
    THINKING = args.thinking
    run_name = f"{args.model}_thinking" if THINKING else args.model
    out_dir = OUT_ROOT / f"itc10s_gemma_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    model, processor = load_model(args.model)
    load_s = round(time.time() - t_load, 1)
    print(f"{MODEL_IDS[args.model]} loaded in {load_s}s on {model.device}", flush=True)
    new = not TIMING_CSV.exists()
    fh = TIMING_CSV.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["model", "file_id", "language", "audio_duration_sec", "segments", "requests", "retries",
                    "input_tokens", "output_tokens", "thinking_tokens", "gen_seconds", "wall_seconds", "sec_per_audio_minute",
                    "aligned", "error", "model_load_seconds"])
    for fid in args.files.split(","):
        src = json.loads((tt.SRC_DIR / f"{fid}.json").read_text())
        segs = [s for s in src.get("segments", []) if s.get("text")]
        stats = {"gen_seconds": 0.0, "requests": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0,
                 "thinking_tokens": 0}
        t0 = time.time()
        err = ""
        try:
            english = translate_call(model, processor, src["language"], segs, stats)
            (out_dir / f"{fid}.json").write_text(json.dumps({
                "file_id": fid, "language": src["language"], "translation_model": MODEL_IDS[args.model],
                "thinking": THINKING,
                "prompt_version": "translate_v1", "seconds": round(time.time() - t0, 1),
                "segments": [{"t": s["t"], "text": s["text"], "en": e} for s, e in zip(segs, english)],
            }, ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001
            err = str(exc)[:200]
        wall = time.time() - t0
        dur = src.get("audio_duration_sec") or 0
        w.writerow([run_name, fid, src["language"], round(dur, 1), len(segs), stats["requests"], stats["retries"],
                    stats["input_tokens"], stats["output_tokens"], stats["thinking_tokens"], round(stats["gen_seconds"], 1), round(wall, 1),
                    round(wall / (dur / 60), 1) if dur else "", not err, err, load_s])
        fh.flush()
        print(f"{run_name} {fid} {src['language']} {round(dur)}s audio, {len(segs)} segs: {round(wall, 1)}s "
              f"({stats['requests']} req, {stats['output_tokens']} out tok){' ERROR ' + err if err else ''}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
