"""Gemini transcription of the call audio with token logging — same instruction
and JSON schema as stage1_transcribe.py (the ground-truth transcripts), but
with a selectable model / thinking budget and a separate output folder, so the
ground truth is never touched.

Output: transcripts_gemini/<out-name>/<file_id>.json with segments, timing,
thinking budget and token usage. Calls already done are skipped.

Usage: python3 transcribe_audio_tokens.py --out-name gemini31fl_think [--model gemini-3.5-flash-lite]
       [--thinking-budget 0] [--workers 6] [--limit N]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import dataset_v2 as dsv2
import gemini_client as gc
from stage1_transcribe import TRANSCRIPTION_INSTRUCTION, TRANSCRIPTION_JSON_SCHEMA


def process(client, rec, args, out_dir):
    t0 = time.time()
    usage = {}
    text = gc.call_with_retries(lambda: gc.generate_text(
        client, args.model, [gc.upload_audio(client, rec.path), TRANSCRIPTION_INSTRUCTION],
        response_json_schema=TRANSCRIPTION_JSON_SCHEMA, usage_out=usage, thinking_budget=args.thinking_budget))
    try:
        segments = gc.parse_json_lenient(text).get("segments", [])
        status = "success"
    except json.JSONDecodeError as exc:
        segments, status = [], f"error: malformed JSON: {exc}"
    out = {"file_id": rec.file_id, "language": rec.language, "model": args.model,
           "thinking_budget": args.thinking_budget, "status": status, "seconds": round(time.time() - t0, 1),
           "usage": usage, "segments": segments}
    (out_dir / f"{rec.file_id}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--thinking-budget", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out_dir = config.PROJECT_ROOT / "transcripts_gemini" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, recs = set(), []
    for r in dsv2.load_dataset_v2(config.DOSTT_DIR):
        if r.file_id not in seen:
            seen.add(r.file_id)
            recs.append(r)

    def done(r):
        f = out_dir / f"{r.file_id}.json"
        return f.exists() and json.loads(f.read_text()).get("status") == "success"
    recs = [r for r in recs if not done(r)][: args.limit]
    print(f"{len(recs)} call(s) to transcribe -> {out_dir}", flush=True)
    client = gc.get_client()
    t0, n, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, client, r, args, out_dir): r for r in recs}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                out = fut.result()
                n += 1
                print(f"[{n}/{len(recs)}] {r.file_id}: {out['status']}, {len(out['segments'])} segments, {out['seconds']}s", flush=True)
                if out["status"] != "success":
                    failed.append(r.file_id)
            except Exception as exc:  # noqa: BLE001
                failed.append(r.file_id)
                print(f"FAILED {r.file_id}: {str(exc)[:200]}", flush=True)
    print(f"Done in {time.time() - t0:.0f}s — ok={n - len(failed)} failed={len(failed)}")


if __name__ == "__main__":
    main()
