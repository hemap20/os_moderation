"""English transcript straight from the call audio with Gemini (no ASR step):
Gemini listens to the full call and writes timestamped English lines.
Mirrors stage1_transcribe.py's request (inline audio, constrained JSON schema,
retries) but translates instead of transcribing verbatim.

Output: translations/audio_gemini31fl/<file_id>.json, with segments [{t, en}],
the model, prompt version and timing. Calls already done are skipped.

Each file also records token usage (summed over all requests for that call,
including resumes) and the thinking budget used.

Usage: python3 translate_audio.py [--workers 6] [--limit N] [--files id1,id2]
       [--model gemini-3.5-flash-lite] [--thinking-budget 0] [--out-name audio_gemini35fl]
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import dataset_v2 as dsv2
import gemini_client as gc

MODEL = "gemini-3.1-flash-lite"
OUT_DIR = config.PROJECT_ROOT / "translations" / "audio_gemini31fl"
PROMPT_VERSION = "audio_translate_v2"
THINKING_BUDGET = 1024  # --thinking-budget; 0 turns thinking off

PROMPT = """
You are a precise audio translation engine. Listen to the ENTIRE attached audio file from start to end
and write an English transcript of it. This is a two-person (sometimes one-person) phone/video call on
"Friends", an app where a user talks to a paid expert. The speakers use {language}, often mixed with
English words.

Rules:
- Cover the full duration of the audio, including trailing seconds. Do not stop early or summarize.
- Break the transcript into short lines, each tagged with a single timestamp "t" in MM:SS format marking
  when that line starts, relative to the start of the audio.
- Write every line in ENGLISH ONLY. Never write {language} or any other Indian language, in any script,
  including romanized {language} in Latin letters (e.g. write "Will you come?", never "Aa jaoge?").
- Translate each line into natural, literal English. Translate faithfully: keep the meaning as said,
  including slang, sexual content, money, phone numbers, app names (WhatsApp, Instagram, Snapchat, etc.)
  and requests to meet. Do not soften, censor, summarise or add explanations.
- Where a speaker already uses an English word, keep that word; translate everything else into English.
- If a line is inaudible/unintelligible, write "[inaudible]" for that line rather than guessing.
- If someone repeats the same filler sound or word many times in a row ("haan haan haan..."), write it
  at most three times followed by "(repeated)". Do not write long repetitions.
- Do not classify, flag, or judge any content. Pure translation only.

Return ONE raw, minified JSON object with this exact shape:
{{"segments": [{{"t": "MM:SS", "en": "..."}}]}}

Do not include any other top-level keys. Your entire response must be only the minified JSON object and
nothing else.
""".strip()

SCHEMA = {
    "type": "object",
    "properties": {"segments": {"type": "array", "items": {
        "type": "object",
        "properties": {"t": {"type": "string"}, "en": {"type": "string"}},
        "required": ["t", "en"]}}},
    "required": ["segments"],
}


def mmss(t):
    m = re.fullmatch(r"\s*(\d+):(\d{1,2})(?::(\d{1,2}))?\s*", t or "")
    if not m:
        return None
    a, b, c = m.groups()
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


SEG_RE = re.compile(r'\{\s*"t"\s*:\s*"([^"]*)"\s*,\s*"en"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')
RESUME = ("\n\nIMPORTANT: lines up to {t} are already done. Translate ONLY the part of the audio that starts "
          "AFTER {t}, through to the end, with timestamps relative to the start of the whole audio.")


def salvage(text):
    """Complete {t, en} objects from a reply whose JSON broke (the model looped
    inside one line until it hit the output limit), plus the timestamp of the
    broken line. The broken line itself is not kept."""
    segs = [{"t": t, "en": json.loads(f'"{en}"')} for t, en in SEG_RE.findall(text)]
    return segs


def request(client, audio, prompt, usage):
    u = {}
    text = gc.call_with_retries(lambda: gc.generate_text(client, MODEL, [audio, prompt], response_json_schema=SCHEMA,
                                                         usage_out=u, thinking_budget=THINKING_BUDGET))
    usage["requests"] += 1
    for k in ("prompt_tokens", "output_tokens", "thinking_tokens"):
        usage[k] += u.get(k, 0)
    for m, n in u.get("prompt_tokens_by_modality", {}).items():
        usage["prompt_tokens_by_modality"][m] = usage["prompt_tokens_by_modality"].get(m, 0) + n
    return text


def process(client, rec):
    t0 = time.time()
    audio = gc.upload_audio(client, rec.path)
    prompt = PROMPT.format(language=rec.language.capitalize())
    segs, resumed = [], 0
    usage = {"requests": 0, "prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "prompt_tokens_by_modality": {}}
    text = request(client, audio, prompt, usage)
    for _ in range(4):
        try:
            part = [s for s in gc.parse_json_lenient(text).get("segments", []) if isinstance(s, dict) and s.get("t")]
            segs += [s for s in part if not segs or (mmss(s["t"]) or 0) > (mmss(segs[-1]["t"]) or 0)]
            break
        except json.JSONDecodeError:
            # The reply looped inside a line until it hit the output limit: keep the
            # complete lines before it, then ask for the rest of the audio only.
            part = salvage(text)
            if not part:
                raise
            segs += [s for s in part if not segs or (mmss(s["t"]) or 0) > (mmss(segs[-1]["t"]) or 0)]
            resumed += 1
            text = request(client, audio, prompt + RESUME.format(t=segs[-1]["t"]), usage)
    else:
        raise ValueError("reply still broken after resuming 4 times")
    from mutagen import File as MutagenFile
    info = MutagenFile(rec.path)
    duration = float(info.info.length) if info is not None and info.info is not None else None
    last = max((mmss(s["t"]) or 0) for s in segs) if segs else 0
    out = {
        "file_id": rec.file_id, "language": rec.language, "audio": str(rec.path.relative_to(config.PROJECT_ROOT)),
        "translation_model": MODEL, "prompt_version": PROMPT_VERSION, "seconds": round(time.time() - t0, 1),
        "audio_duration_sec": duration, "last_timestamp_sec": last,
        # Same incomplete-transcript rule as stage1_transcribe.py: the last line starts well before the end.
        "possibly_incomplete": bool(duration and duration - last > max(config.INCOMPLETE_TRANSCRIPT_MIN_GAP_SEC,
                                                                        config.INCOMPLETE_TRANSCRIPT_GAP_FRACTION * duration)),
        "resumed_after_loop": resumed,
        "thinking_budget": THINKING_BUDGET, "usage": usage,
        "segments": [{"t": s["t"], "en": s.get("en", "")} for s in segs],
    }
    (OUT_DIR / f"{rec.file_id}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main():
    global MODEL, OUT_DIR, THINKING_BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--files", type=str, default=None, help="comma-separated file_ids")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--thinking-budget", type=int, default=THINKING_BUDGET)
    ap.add_argument("--out-name", default=None, help="folder under translations/ (default audio_gemini31fl)")
    args = ap.parse_args()
    MODEL, THINKING_BUDGET = args.model, args.thinking_budget
    if args.out_name:
        OUT_DIR = config.PROJECT_ROOT / "translations" / args.out_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen, recs = set(), []
    for r in dsv2.load_dataset_v2(config.DOSTT_DIR):
        if r.file_id not in seen:
            seen.add(r.file_id)
            recs.append(r)
    if args.files:
        want = set(args.files.split(","))
        recs = [r for r in recs if r.file_id in want]
    recs = [r for r in recs if not (OUT_DIR / f"{r.file_id}.json").exists()]
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} call(s) to translate from audio -> {OUT_DIR}", flush=True)
    client = gc.get_client()
    done, failed, lock = 0, [], threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, client, r): r for r in recs}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                out = fut.result()
                with lock:
                    done += 1
                flag = " POSSIBLY INCOMPLETE" if out["possibly_incomplete"] else ""
                print(f"[{done}/{len(recs)}] {r.file_id}: {len(out['segments'])} lines, "
                      f"{out['audio_duration_sec'] and round(out['audio_duration_sec'])}s audio, {out['seconds']}s{flag}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append((r.file_id, str(exc)[:200]))
                print(f"FAILED {r.file_id}: {str(exc)[:200]}", flush=True)
    print(f"Done in {time.time() - t0:.0f}s — translated={done} failed={len(failed)}")
    for fid, err in failed:
        print(f"  failed: {fid}: {err}")


if __name__ == "__main__":
    main()
