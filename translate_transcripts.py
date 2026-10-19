"""Translate ASR transcripts into English with Gemini, segment by segment,
keeping every segment's timestamp so English windows (e.g. +-10 s) can be cut
exactly like the original-language ones.

Input:  indic_transcribe_core_results/chunk_10s/transcripts/<file_id>.json
Output: translations/itc10s_gemini31fl/<file_id>.json  -- the original segments
        with an added "en" field, plus model/prompt metadata. Calls already
        translated are skipped, so re-running resumes.

Each request carries at most CHUNK segments; a long call is split into several
requests and stitched back in order. A response whose segment count or
timestamps don't match its input is retried, then re-split into smaller
chunks; anything still failing is listed in the summary.

Usage: python3 translate_transcripts.py [--workers 8] [--limit N]
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
try:
    import gemini_client as gc
except ImportError:  # local_translate.py reuses this module's prompt from the Gemma venv
    gc = None

MODEL = "gemini-3.1-flash-lite"
SRC_DIR = config.PROJECT_ROOT / "indic_transcribe_core_results" / "chunk_10s" / "transcripts"
OUT_DIR = config.PROJECT_ROOT / "translations" / "itc10s_gemini31fl"
CHUNK = 60

PROMPT = """You are translating a phone call transcript from {language} into English.

The transcript comes from automatic speech recognition (ASR) of a call on "Friends", an app where a
user talks to a paid expert. It is noisy: words may be misspelled, split, merged or misheard, and
speakers often mix {language} with English words (code-mixing).

Translate each segment into natural, literal English.

Rules:
1. Translate every segment, in order. Return exactly one output segment per input segment, with the
   SAME "t" value. Never merge, split, skip or reorder segments.
2. Translate faithfully. Keep the meaning as said, including slang, sexual content, money,
   phone numbers, app names (WhatsApp, Instagram, Snapchat, etc.) and requests to meet. Do not
   soften, censor, summarise or add explanations.
3. English words already in the text: keep them as they are.
4. If part of a segment is garbled or unintelligible, translate what you can and write [unclear]
   for the rest. If a whole segment is unintelligible, output "[unclear]".
5. Do not guess meaning that isn't there. Do not add speaker labels or punctuation-only changes.

Input segments (JSON):
{segments_json}

Return ONLY a JSON object of this form:
{{"segments": [{{"t": "MM:SS", "en": "<English translation>"}}, ...]}}"""

SCHEMA = {
    "type": "object",
    "properties": {"segments": {"type": "array", "items": {
        "type": "object",
        "properties": {"t": {"type": "string"}, "en": {"type": "string"}},
        "required": ["t", "en"]}}},
    "required": ["segments"],
}


def collapse_repeats(text: str, keep: int = 3, min_run: int = 8, max_ngram: int = 4) -> str:
    """ASR sometimes gets stuck and repeats one word or a short phrase hundreds of
    times ("जी जी जी ...", "பதினொன்னு மணிக்கு பதினொன்னு மணிக்கு ..."), which makes the
    translation reply loop until its JSON breaks. Collapse a run of >= min_run
    repeats of any 1..max_ngram-word phrase to `keep` copies plus an [xN] marker."""
    words, out, i = text.split(), [], 0
    while i < len(words):
        best = None
        for n in range(1, max_ngram + 1):
            unit = words[i:i + n]
            if len(unit) < n:
                break
            run = 1
            while words[i + run * n:i + (run + 1) * n] == unit:
                run += 1
            if run >= min_run and (best is None or run * n > best[0] * best[1]):
                best = (run, n)
        if best:
            run, n = best
            out += words[i:i + keep * n] + [f"[x{run}]"]
            i += run * n
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def translate_chunk(client, language, segs):
    """One request for a list of {t, text}; returns the list of English strings,
    or raises ValueError if the response doesn't line up with the input."""
    payload = json.dumps([{"t": s["t"], "text": collapse_repeats(s["text"])} for s in segs], ensure_ascii=False)
    prompt = PROMPT.format(language=language.capitalize(), segments_json=payload)
    text = gc.call_with_retries(lambda: gc.generate_text(client, MODEL, [prompt], response_json_schema=SCHEMA))
    try:
        out = gc.parse_json_lenient(text).get("segments", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON reply: {exc}")
    if len(out) != len(segs):
        raise ValueError(f"segment count {len(out)} != {len(segs)}")
    bad = [i for i, (a, b) in enumerate(zip(segs, out)) if a["t"] != b.get("t")]
    if bad:
        raise ValueError(f"{len(bad)} timestamp mismatch(es), first at index {bad[0]}")
    return [o.get("en", "") for o in out]


def translate_segments(client, language, segs, size=CHUNK, depth=0):
    """Translate in chunks of `size`; a chunk that fails validation twice is
    re-split in half (down to single segments) before giving up."""
    english = []
    for i in range(0, len(segs), size):
        part = segs[i:i + size]
        last = None
        for _ in range(2):
            try:
                english += translate_chunk(client, language, part)
                break
            except ValueError as exc:
                last = exc
        else:
            if len(part) == 1 or depth >= 6:
                raise ValueError(f"could not align chunk at segment {i}: {last}")
            english += translate_segments(client, language, part, max(1, len(part) // 2), depth + 1)
    return english


def process(client, path: Path):
    d = json.loads(path.read_text())
    segs = [s for s in d.get("segments", []) if s.get("text")]
    t0 = time.time()
    english = translate_segments(client, d.get("language", "the original language"), segs) if segs else []
    out = {
        "file_id": d["file_id"], "language": d.get("language"), "source": str(path.relative_to(config.PROJECT_ROOT)),
        "translation_model": MODEL, "prompt_version": "translate_v1", "seconds": round(time.time() - t0, 1),
        "segments": [{"t": s["t"], "text": s["text"], "en": e,
                      **({"collapsed_repeats": True} if collapse_repeats(s["text"]) != s["text"] else {})}
                     for s, e in zip(segs, english)],
    }
    (OUT_DIR / path.name).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return len(segs), out["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(SRC_DIR.glob("*.json")) if not (OUT_DIR / p.name).exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} call(s) to translate -> {OUT_DIR}", flush=True)
    client = gc.get_client()
    done, failed, lock = 0, [], threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, client, p): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                n, secs = fut.result()
                with lock:
                    done += 1
                print(f"[{done}/{len(todo)}] {p.stem}: {n} segments in {secs}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append((p.stem, str(exc)[:200]))
                print(f"FAILED {p.stem}: {str(exc)[:200]}", flush=True)
    print(f"Done in {time.time() - t0:.0f}s — translated={done} failed={len(failed)}")
    for fid, err in failed:
        print(f"  failed: {fid}: {err}")


if __name__ == "__main__":
    main()
