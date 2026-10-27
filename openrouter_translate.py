"""Translate the Indic-Transcribe-Core 10 s transcripts into English with Gemma 4
31B through OpenRouter (thinking or not), timing every call.

Same prompt, chunking, repetition collapsing and alignment checks as
translate_transcripts.py (the Gemini text translation), so the outputs compare
directly. Providers are pinned as in the 31B moderation runs (Parasail for
non-thinking, CoreWeave for thinking) for consistent timings.

Output: translations/itc10s_gemma31b[_thinking]/<file_id>.json (same shape as the
Gemini translations, plus timing) and translations/openrouter_translation_timing.csv.

Usage: python3 openrouter_translate.py [--thinking] [--workers 4] [--limit N] [--files a,b]
"""
import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config
import translate_transcripts as tt
from openrouter_text import API_URL, api_key

MODEL_ID = "google/gemma-4-31b-it"
OUT_ROOT = config.PROJECT_ROOT / "translations"
TIMING_CSV = OUT_ROOT / "openrouter_translation_timing.csv"
COLS = ["run", "file_id", "language", "audio_duration_sec", "segments", "requests", "retries", "request_seconds",
        "wall_seconds", "rtf", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd", "aligned", "error"]


def request(session, prompt, thinking, provider, stats):
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": 16384 if thinking else 8192, "reasoning": {"enabled": thinking},
            "usage": {"include": True}, "provider": {"order": [provider], "allow_fallbacks": False}}
    last = None
    for attempt in range(1, 6):
        t0 = time.perf_counter()
        try:
            r = session.post(API_URL, json=body, timeout=900)
            if r.status_code == 200:
                d = r.json()
                if "error" not in d:
                    stats["request_seconds"] += time.perf_counter() - t0
                    u = d.get("usage") or {}
                    stats["prompt_tokens"] += u.get("prompt_tokens") or 0
                    stats["completion_tokens"] += u.get("completion_tokens") or 0
                    stats["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                    stats["cost_usd"] += u.get("cost") or 0
                    return d["choices"][0]["message"].get("content") or ""
                last = d["error"]
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in (408, 429, 500, 502, 503, 504):
                    break
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise RuntimeError(f"OpenRouter failed: {last}")


def translate_chunk(session, language, segs, thinking, provider, stats):
    payload = json.dumps([{"t": s["t"], "text": tt.collapse_repeats(s["text"])} for s in segs], ensure_ascii=False)
    text = request(session, tt.PROMPT.format(language=language.capitalize(), segments_json=payload), thinking, provider, stats)
    stats["requests"] += 1
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


def translate_call(session, language, segs, thinking, provider, stats, size=tt.CHUNK, depth=0):
    english = []
    for i in range(0, len(segs), size):
        part = segs[i:i + size]
        last = None
        for _ in range(2):
            try:
                english += translate_chunk(session, language, part, thinking, provider, stats)
                break
            except ValueError as exc:
                last = exc
                stats["retries"] += 1
        else:
            if len(part) == 1 or depth >= 6:
                raise ValueError(f"could not align chunk at segment {i}: {last}")
            english += translate_call(session, language, part, thinking, provider, stats, max(1, len(part) // 2), depth + 1)
    return english


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--files", default=None)
    args = ap.parse_args()
    run = "gemma31b_thinking" if args.thinking else "gemma31b"
    provider = "CoreWeave" if args.thinking else "Parasail"
    out_dir = OUT_ROOT / f"itc10s_{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(tt.SRC_DIR.glob("*.json"))
    if args.files:
        want = set(args.files.split(","))
        paths = [p for p in paths if p.stem in want]
    paths = [p for p in paths if not (out_dir / p.name).exists()]
    if args.limit:
        paths = paths[: args.limit]
    print(f"{run} via {provider}: {len(paths)} call(s)", flush=True)

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "X-Title": "os_moderation"})
    lock = threading.Lock()
    new = not TIMING_CSV.exists()
    fh = TIMING_CSV.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    def work(p):
        src = json.loads(p.read_text())
        segs = [s for s in src.get("segments", []) if s.get("text")]
        stats = {"requests": 0, "retries": 0, "request_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                 "reasoning_tokens": 0, "cost_usd": 0.0}
        t0 = time.time()
        err = ""
        try:
            english = translate_call(session, src["language"], segs, args.thinking, provider, stats) if segs else []
            (out_dir / p.name).write_text(json.dumps({
                "file_id": src["file_id"], "language": src["language"], "translation_model": MODEL_ID, "thinking": args.thinking,
                "provider": provider, "prompt_version": "translate_v1", "seconds": round(time.time() - t0, 1),
                "segments": [{"t": s["t"], "text": s["text"], "en": e} for s, e in zip(segs, english)],
            }, ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001
            err = str(exc)[:200]
        wall = time.time() - t0
        dur = src.get("audio_duration_sec") or 0
        row = {"run": run, "file_id": src["file_id"], "language": src["language"], "audio_duration_sec": round(dur, 1),
               "segments": len(segs), "requests": stats["requests"], "retries": stats["retries"],
               "request_seconds": round(stats["request_seconds"], 2), "wall_seconds": round(wall, 2),
               "rtf": round(stats["request_seconds"] / dur, 4) if dur else "", "prompt_tokens": stats["prompt_tokens"],
               "completion_tokens": stats["completion_tokens"], "reasoning_tokens": stats["reasoning_tokens"],
               "cost_usd": round(stats["cost_usd"], 6), "aligned": not err, "error": err}
        with lock:
            w.writerow(row)
            fh.flush()
        return row

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, p) for p in paths]):
            r = fut.result()
            done += 1
            print(f"[{done}/{len(paths)}] {r['file_id']} {r['language']}: {r['request_seconds']}s for {r['audio_duration_sec']}s audio"
                  f"{' ERROR ' + r['error'] if r['error'] else ''}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
