"""Transcript-based moderation through OpenRouter — e.g. Gemma 4 31B, which
is too large to run locally. Same input as gemma_local_text.py (the
Indic-Conformer transcript as '[t] text' lines, same prompt + JSON schema)
and the same output format (GemmaFileResult under gemma_results*/), so
analyze_results.py / threshold_analysis.py score it like any other model.

Per-file timing is saved in each result JSON and collected into
<output_dir>/inference_times.csv:
  - inference_time_sec: wall-clock seconds of the successful request
    (network + provider queueing + generation), measured here.
  - inference_meta: OpenRouter's own numbers for that generation —
    latency_ms (time to first token), generation_time_ms, token counts
    (incl. reasoning), total cost in USD, and which provider served it.

Confidence/entropy: OpenRouter returns at most the top-20 logprobs per
token, so excerpt entropy is computed over those 20 (renormalized) instead
of the full vocabulary as local runs do — comparable in ranking, but not
numerically identical to local entropy. Logprob-derived confidence uses the
chosen token's exact logprob, same as local. provider.require_parameters
routes only to providers that actually return logprobs.

Needs OPENROUTER_API_KEY (or OPEN_ROUTER) in .env / the environment.

Usage:
    python3 openrouter_text.py --prompt-path prompt_v7.py --dataset-root Dostt_dev --results-tag promptv7 --dry-run
    python3 openrouter_text.py --prompt-path prompt_v7.py --dataset-root Dostt_dev --results-tag promptv7
    python3 openrouter_text.py --prompt-path prompt_v7.py --results-tag promptv7 --thinking --workers 8
"""
import argparse
import csv
import json
import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

import config
import dataset_v2 as dsv2
import prompt_loader
from gemma_local import parse_and_score_flags
from pipeline_logging import StageLogger
from schemas_gemma import GemmaFileResult

API_URL = "https://openrouter.ai/api/v1/chat/completions"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
TOP_LOGPROBS = 20  # OpenRouter/OpenAI maximum
MAX_ATTEMPTS = 5

DATASET_DIR: Path = config.DOSTT_DIR
GEMMA_RESULTS_DIR: Path = config.PROJECT_ROOT / "gemma_results"
PROMPT_PATH: Path = config.CLASSIFICATION_PROMPT_PATH
RAW_SCHEMA_STR: str = ""


# --transcript-source -> (results dir under the repo, short tag used in output dir names)
TRANSCRIPT_SOURCES = {
    "conformer": ("indic_conformer_results", "conformer"),
    "itc": ("indic_transcribe_core_results", "itc"),
}


def output_dir(model_key: str, source: str, transcript_chunk_seconds: float, thinking: bool) -> Path:
    tag = TRANSCRIPT_SOURCES[source][1]
    return GEMMA_RESULTS_DIR / f"{model_key}_{tag}{transcript_chunk_seconds:g}s_text_{'thinking' if thinking else 'nothinking'}"


def load_transcript_lines(source: str, transcript_chunk_seconds: float, file_id: str) -> Optional[str]:
    """'[t] text' per line, same format gemma_local_text.py feeds the model.
    Both ASR pipelines write the same per-file JSON shape."""
    p = (config.PROJECT_ROOT / TRANSCRIPT_SOURCES[source][0] / f"chunk_{transcript_chunk_seconds:g}s"
         / "transcripts" / f"{file_id}.json")
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status", "success") != "success":
        return None
    segments = data.get("segments", [])
    if not segments:
        return data.get("full_text", "")
    return "\n".join(f"[{s.get('t')}] {s.get('text', '')}" for s in segments)


def api_key() -> str:
    names = ("OPENROUTER_API_KEY", "OPEN_ROUTER")
    key = next((os.environ[n] for n in names if os.environ.get(n)), None)
    if not key:
        env = config.PROJECT_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                name, _, value = line.strip().partition("=")
                if name.strip() in names and value.strip():
                    key = value.strip().strip('"').strip("'")
                    break
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set — add it to .env or export it.")
    return key


def to_token_infos(logprobs_content: list) -> list:
    """OpenRouter logprobs -> the {token, logprob, entropy, topk} dicts that
    gemma_local.parse_and_score_flags expects. Entropy is over the returned
    top-k only (renormalized), see module docstring."""
    out = []
    for item in logprobs_content or []:
        topk = [{"token": t.get("token", ""), "logprob": t.get("logprob")} for t in item.get("top_logprobs") or []
                if t.get("logprob") is not None]
        entropy = None
        if topk:
            ps = [math.exp(t["logprob"]) for t in topk]
            z = sum(ps)
            entropy = -sum((p / z) * math.log(p / z) for p in ps if p > 0)
        out.append({"token": item.get("token", ""), "token_id": None, "logprob": item.get("logprob"),
                    "entropy": entropy, "topk": topk})
    return out


def call_openrouter(session: requests.Session, model_id: str, transcript_text: str, thinking: bool,
                    max_tokens: int, logger: StageLogger, file_id: str, want_logprobs: bool = True,
                    provider: Optional[str] = None) -> tuple:
    prompt_text = prompt_loader.render_prompt(PROMPT_PATH, json_schema_str=RAW_SCHEMA_STR)
    transcript_part = (
        "[TRANSCRIPT OF THE CALL — analyse this text, which is the full "
        "transcription of the audio, in place of the audio itself]\n\n"
        f"{transcript_text}"
    )
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": transcript_part},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": thinking},
        "usage": {"include": True},
    }
    routing = {}
    if want_logprobs:
        body.update({"logprobs": True, "top_logprobs": TOP_LOGPROBS})
        routing["require_parameters"] = True
    if provider:
        # Pin one provider: consistent hardware/quantization makes per-file
        # timings comparable, and some providers silently drop logprobs.
        routing.update({"order": [provider], "allow_fallbacks": False})
    if routing:
        body["provider"] = routing
    total_t0 = time.perf_counter()
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            resp = session.post(API_URL, json=body, timeout=600)
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"API error: {data['error']}")
                return data, elapsed, time.perf_counter() - total_t0, attempt
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in (408, 429, 500, 502, 503, 504):
                break
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_err = str(exc)
        wait = min(60, 5 * 2 ** (attempt - 1))
        logger.warn(f"{file_id}: attempt {attempt} failed ({last_err}); retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"OpenRouter request failed after {attempt} attempt(s): {last_err}")


def fetch_generation_stats(session: requests.Session, gen_id: Optional[str]) -> dict:
    """OpenRouter's own record of the generation (provider timing, native
    token counts, cost). Available a moment after completion, so poll briefly."""
    if not gen_id:
        return {}
    for _ in range(15):
        try:
            r = session.get(GENERATION_URL, params={"id": gen_id}, timeout=30)
            if r.status_code == 200:
                d = r.json().get("data", {})
                return {
                    "generation_id": gen_id,
                    "provider": d.get("provider_name"),
                    "latency_ms": d.get("latency"),
                    "generation_time_ms": d.get("generation_time"),
                    "tokens_prompt": d.get("native_tokens_prompt"),
                    "tokens_completion": d.get("native_tokens_completion"),
                    "tokens_reasoning": d.get("native_tokens_reasoning"),
                    "cost_usd": d.get("total_cost"),
                    "finish_reason": d.get("native_finish_reason") or d.get("finish_reason"),
                }
        except requests.RequestException:
            pass
        time.sleep(4)
    return {"generation_id": gen_id}


def process_file(session, record, args, logger) -> Optional[tuple]:
    transcript_text = load_transcript_lines(args.transcript_source, args.transcript_chunk_seconds, record.file_id)
    if transcript_text is None:
        return None
    data, request_sec, total_sec, attempts = call_openrouter(
        session, args.model_id, transcript_text, args.thinking, args.max_tokens, logger, record.file_id,
        want_logprobs=not args.no_logprobs, provider=args.provider)
    choice = data["choices"][0]
    message = choice.get("message") or {}
    answer_text = message.get("content") or ""
    token_infos = to_token_infos((choice.get("logprobs") or {}).get("content"))
    flags, missing = parse_and_score_flags(answer_text, token_infos, chunk_offset_sec=0.0, logger=logger)

    usage = data.get("usage") or {}
    meta = {
        "model_id": args.model_id,
        "transcript_source": TRANSCRIPT_SOURCES[args.transcript_source][0],
        "transcript_chunk_seconds": args.transcript_chunk_seconds,
        "transcript_chars": len(transcript_text),
        "request_attempts": attempts,
        "total_time_incl_retries_sec": round(total_sec, 3),
        "logprobs_returned": bool(token_infos),
        "finish_reason": choice.get("finish_reason"),
        "tokens_prompt": usage.get("prompt_tokens"),
        "tokens_completion": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
    }
    # OpenRouter's own stats (provider timing, cost) are fetched later in a
    # background thread — see main() — so the next file isn't held up.
    if not token_infos:
        logger.warn(f"{record.file_id}: provider returned no logprobs — logprob confidence/entropy will be empty")

    result = GemmaFileResult(
        file_id=record.file_id, model=args.model_key, thinking=args.thinking,
        chunk_seconds=args.transcript_chunk_seconds, flags=flags, chunks_total=1, chunks_failed=0,
        status="success", missing_field_counts=missing,
        inference_time_sec=round(request_sec, 3), inference_meta=meta,
    )
    raw = {"content": answer_text, "reasoning": message.get("reasoning"), "response": data}
    return result, raw, data.get("id")


def write_timing_csv(out_dir: Path) -> Path:
    path = out_dir / "inference_times.csv"
    cols = ["file_id", "status", "inference_time_sec", "transcript_chars", "latency_ms", "generation_time_ms", "tokens_prompt",
            "tokens_completion", "tokens_reasoning", "cost_usd", "provider", "request_attempts", "n_flags"]
    rows = []
    for f in sorted((out_dir / "results").glob("*.json")):
        d = json.loads(f.read_text())
        m = d.get("inference_meta") or {}
        rows.append({"file_id": d["file_id"], "status": d.get("status"), "inference_time_sec": d.get("inference_time_sec"),
                     "n_flags": len(d.get("flags") or []), **{k: m.get(k) for k in cols if k in m}})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    global DATASET_DIR, GEMMA_RESULTS_DIR, PROMPT_PATH, RAW_SCHEMA_STR

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-31b-it", help="OpenRouter model slug")
    parser.add_argument("--model-key", default="31b", help="Short name used in the output dir / result 'model' field")
    parser.add_argument("--thinking", action="store_true", help="Enable the model's reasoning mode")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Default 4096 non-thinking, 16384 thinking (reasoning tokens count toward it)")
    parser.add_argument("--provider", default=None,
                        help="Pin one OpenRouter provider, e.g. Novita or CoreWeave (no fallbacks). Recommended: "
                             "consistent timings, and some providers (e.g. Venice) silently omit logprobs.")
    parser.add_argument("--no-logprobs", action="store_true",
                        help="Don't request logprobs (use if OpenRouter says no endpoint supports the requested "
                             "parameters). Self-reported confidence still works; logprob confidence/entropy are empty.")
    parser.add_argument("--transcript-source", choices=sorted(TRANSCRIPT_SOURCES), default="conformer",
                        help="conformer = indic_conformer_results, itc = indic_transcribe_core_results")
    parser.add_argument("--transcript-chunk-seconds", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--results-tag", type=str, default=None)
    parser.add_argument("--prompt-path", type=str, default=str(config.CLASSIFICATION_PROMPT_PATH))
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 16384 if args.thinking else 4096

    paths = config.dataset_paths(args.dataset_root, args.results_tag)
    DATASET_DIR = paths["dataset_dir"]
    GEMMA_RESULTS_DIR = paths["gemma_results_dir"]
    p = Path(args.prompt_path)
    PROMPT_PATH = p if p.is_absolute() else config.PROJECT_ROOT / p
    RAW_SCHEMA_STR = json.dumps(prompt_loader.schema_module_for_prompt(PROMPT_PATH).raw_model_output_json_schema())

    mode = "thinking" if args.thinking else "nothinking"
    logger = StageLogger(f"openrouter_text_{args.model_key}_{args.transcript_source}_{mode}")
    out_dir = output_dir(args.model_key, args.transcript_source, args.transcript_chunk_seconds, args.thinking)
    logger.info(f"{args.model_id} ({mode}), prompt={PROMPT_PATH.name}, out={out_dir}")

    seen, records = set(), []
    for r in dsv2.load_dataset_v2(DATASET_DIR):
        if r.file_id not in seen:
            seen.add(r.file_id)
            records.append(r)
    if args.dry_run:
        records = records[: args.dry_run_limit]
    elif not args.force:
        def done(fid):
            f = out_dir / "results" / f"{fid}.json"
            try:
                return f.exists() and json.loads(f.read_text()).get("status") == "success"
            except Exception:
                return False
        before = len(records)
        records = [r for r in records if not done(r.file_id)]
        logger.info(f"Skipping {before - len(records)} already-done file(s)")
    if args.limit:
        records = records[: args.limit]

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                            "X-Title": "os_moderation"})
    lock = threading.Lock()
    counts = {"success": 0, "error": 0, "skipped": 0}
    t0 = time.time()

    def work(i, record):
        try:
            out = process_file(session, record, args, logger)
            if out is None:
                logger.warn(f"{record.file_id}: no {args.transcript_source} transcript, skipping")
                with lock:
                    counts["skipped"] += 1
                return
            result, raw, gen_id = out
            if args.dry_run:
                result.inference_meta.update({k: v for k, v in fetch_generation_stats(session, gen_id).items() if v is not None})
                print(result.model_dump_json(indent=2))
            else:
                result_path = out_dir / "results" / f"{record.file_id}.json"
                (out_dir / "results").mkdir(parents=True, exist_ok=True)
                (out_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
                result_path.write_text(result.model_dump_json(indent=2))
                (out_dir / "raw_responses" / f"{record.file_id}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2))

                def add_stats(result=result, result_path=result_path, gen_id=gen_id):
                    result.inference_meta.update({k: v for k, v in fetch_generation_stats(session, gen_id).items() if v is not None})
                    result_path.write_text(result.model_dump_json(indent=2))
                stats_pool.submit(add_stats)
            with lock:
                counts["success"] += 1
            logger.info(f"[{i}/{len(records)}] {record.file_id}: {len(result.flags)} flag(s), "
                        f"{result.inference_time_sec}s, ${result.inference_meta.get('cost_usd')}")
        except Exception as exc:  # noqa: BLE001
            with lock:
                counts["error"] += 1
            logger.error(f"{record.file_id}: FAILED — {exc}\n{traceback.format_exc()}")
            if not args.dry_run:
                (out_dir / "results").mkdir(parents=True, exist_ok=True)
                err = GemmaFileResult(file_id=record.file_id, model=args.model_key, thinking=args.thinking,
                                      chunk_seconds=args.transcript_chunk_seconds, status="error", error=str(exc))
                (out_dir / "results" / f"{record.file_id}.json").write_text(err.model_dump_json(indent=2))

    stats_pool = ThreadPoolExecutor(max_workers=4)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(work, i, r) for i, r in enumerate(records, 1)]):
            fut.result()
    stats_pool.shutdown(wait=True)  # every result file has its OpenRouter stats before the CSV is built

    if not args.dry_run and (out_dir / "results").exists():
        logger.info(f"Per-file timings: {write_timing_csv(out_dir)}")
    logger.info(f"Done in {time.time() - t0:.1f}s — success={counts['success']} error={counts['error']} "
                f"skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
