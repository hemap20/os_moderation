"""Semantic WER — an LLM-judged alternative to asr_metrics.word_error_rate()
that only counts MEANING-CHANGING discrepancies between the Gemini
reference transcript and an Indic-Conformer hypothesis, ignoring
phonetic/spelling variants that don't actually change what was said (e.g.
"व्हाट्सएप" vs "वाट्सएप" — same word, different spelling).

Standard WER treats every substitution as equally wrong; that's the wrong
lens for judging whether a transcript is good enough to classify from —
what actually matters is whether numbers, negations, names, and
policy-relevant content survive, not exact spelling.

Uses the main .venv's Gemini client (google-genai) — this is plain text-in
text-out, no audio, no local ML weights needed.

Usage:
    python3 semantic_wer.py --dry-run --dry-run-limit 3
    python3 semantic_wer.py --chunk-seconds 2,5,7,10
    python3 semantic_wer.py --force --limit 20
"""
import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

import config
import dataset_v2 as dsv2
import gemini_client
from pipeline_logging import StageLogger

SEMANTIC_WER_INSTRUCTION = """
You are comparing a REFERENCE transcript (assumed accurate) against a
HYPOTHESIS transcript (from a different, less accurate speech recognizer)
of the SAME audio, both in the same native script/language.

Identify ONLY discrepancies where the hypothesis changes the MEANING of
what was said. Explicitly IGNORE:
- Spelling/phonetic variants of the same word (e.g. different ways of
  writing a loanword like "WhatsApp" in native script)
- Minor filler word differences ("um", repeated words, hesitations)
- Punctuation-only differences
- Word order that doesn't change meaning

DO flag discrepancies where:
- A number, ID, phone number, or username is altered or garbled
- A negation is added, dropped, or flipped ("not" missing/added)
- A named entity (person, place, app/platform name) is changed to a
  different one (not just respelled)
- Content present in the reference is entirely missing from the
  hypothesis in a way that loses meaning (not just a filler word)
- The hypothesis says something that materially contradicts or
  misrepresents the reference

For each such discrepancy, report:
- reference_text: the exact reference-transcript span affected
- hypothesis_text: what the hypothesis said instead (empty string if the
  content is simply missing from the hypothesis)
- category: one of "number_or_id_altered", "negation_flipped",
  "named_entity_altered", "key_content_missing", "other_meaning_change"
- word_count_affected: how many words in the REFERENCE span are affected
  by this specific discrepancy (count reference words, not hypothesis words)
- explanation: one short sentence

Return ONE raw, minified JSON object with this exact shape:
{"discrepancies": [{"reference_text": "...", "hypothesis_text": "...", "category": "...", "word_count_affected": 0, "explanation": "..."}]}

If there are no meaning-changing discrepancies, return {"discrepancies": []}.
Your entire response must be only the minified JSON object and nothing else.
""".strip()

SEMANTIC_WER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "discrepancies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference_text": {"type": "string"},
                    "hypothesis_text": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "number_or_id_altered",
                            "negation_flipped",
                            "named_entity_altered",
                            "key_content_missing",
                            "other_meaning_change",
                        ],
                    },
                    "word_count_affected": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["reference_text", "hypothesis_text", "category", "word_count_affected", "explanation"],
            },
        },
    },
    "required": ["discrepancies"],
}


class Discrepancy(BaseModel):
    reference_text: str
    hypothesis_text: str
    category: Literal["number_or_id_altered", "negation_flipped", "named_entity_altered", "key_content_missing", "other_meaning_change"]
    word_count_affected: int
    explanation: str


class SemanticWERResult(BaseModel):
    file_id: str
    chunk_seconds: float
    reference_word_count: int
    affected_word_count: int
    semantic_wer: Optional[float]
    discrepancies: List[Discrepancy] = Field(default_factory=list)
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None


SOURCE_DIRS = {
    "indic_conformer": "indic_conformer_results",
    "indic_transcribe_core": "indic_transcribe_core_results",
}


def output_dir(source: str, chunk_seconds: float) -> Path:
    return config.PROJECT_ROOT / "semantic_wer_results" / source / f"chunk_{chunk_seconds:g}s"


def load_reference(record: dsv2.FileRecordV2) -> Optional[str]:
    p = record.transcripts_dir / f"{record.file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    return data["transcript"].get("full_text", "")


def load_hypothesis(source: str, chunk_seconds: float, file_id: str) -> Optional[str]:
    p = config.PROJECT_ROOT / SOURCE_DIRS[source] / f"chunk_{chunk_seconds:g}s" / "transcripts" / f"{file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    return data.get("full_text", "")


def compute_semantic_wer(client, file_id: str, chunk_seconds: float, reference: str, hypothesis: str, logger: StageLogger) -> SemanticWERResult:
    ref_word_count = len(reference.split())

    if not reference.strip():
        return SemanticWERResult(
            file_id=file_id, chunk_seconds=chunk_seconds, reference_word_count=0,
            affected_word_count=0, semantic_wer=None, status="success",
        )

    prompt_content = f"{SEMANTIC_WER_INSTRUCTION}\n\n[REFERENCE]\n{reference}\n\n[HYPOTHESIS]\n{hypothesis}"

    def do_call():
        return gemini_client.generate_text(
            client, config.GEMINI_MODEL, contents=[prompt_content],
            response_json_schema=SEMANTIC_WER_JSON_SCHEMA,
        )

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"{file_id} chunk={chunk_seconds}s: attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
    parsed = gemini_client.parse_json_lenient(raw_text)
    discrepancies = [Discrepancy(**d) for d in parsed.get("discrepancies", [])]

    affected = sum(d.word_count_affected for d in discrepancies)
    semantic_wer = affected / ref_word_count if ref_word_count else None

    return SemanticWERResult(
        file_id=file_id, chunk_seconds=chunk_seconds, reference_word_count=ref_word_count,
        affected_word_count=affected, semantic_wer=semantic_wer, discrepancies=discrepancies, status="success",
    )


BATCH_INSTRUCTION = """
You are comparing several (REFERENCE, HYPOTHESIS) transcript pairs of
DIFFERENT audio calls, each identified by a file_id. Each pair has the
same REFERENCE-vs-HYPOTHESIS relationship: the REFERENCE is assumed
accurate; the HYPOTHESIS comes from a different, less accurate speech
recognizer of the SAME audio, in the same native script/language.

For EACH pair independently, identify ONLY discrepancies where the
hypothesis changes the MEANING of what was said. Explicitly IGNORE:
- Spelling/phonetic variants of the same word (e.g. different ways of
  writing a loanword like "WhatsApp" in native script)
- Minor filler word differences ("um", repeated words, hesitations)
- Punctuation-only differences
- Word order that doesn't change meaning

DO flag discrepancies where:
- A number, ID, phone number, or username is altered or garbled
- A negation is added, dropped, or flipped ("not" missing/added)
- A named entity (person, place, app/platform name) is changed to a
  different one (not just respelled)
- Content present in the reference is entirely missing from the
  hypothesis in a way that loses meaning (not just a filler word)
- The hypothesis says something that materially contradicts or
  misrepresents the reference

For each discrepancy, report: reference_text, hypothesis_text, category
(one of "number_or_id_altered", "negation_flipped", "named_entity_altered",
"key_content_missing", "other_meaning_change"), word_count_affected (count
of REFERENCE words affected), explanation (one short sentence).

Return ONE raw, minified JSON object with this exact shape, one entry per
input pair (use the SAME file_id given, and include every file_id even if
its discrepancies list is empty):
{"results": [{"file_id": "...", "discrepancies": [{"reference_text": "...", "hypothesis_text": "...", "category": "...", "word_count_affected": 0, "explanation": "..."}]}]}

Your entire response must be only the minified JSON object and nothing else.
""".strip()

BATCH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                    "discrepancies": SEMANTIC_WER_JSON_SCHEMA["properties"]["discrepancies"],
                },
                "required": ["file_id", "discrepancies"],
            },
        },
    },
    "required": ["results"],
}


def compute_semantic_wer_batch(client, chunk_seconds: float, items: List[dict], logger: StageLogger) -> List[SemanticWERResult]:
    """items: list of {file_id, reference, hypothesis}. Returns one SemanticWERResult per item, same order."""
    scoreable = [it for it in items if it["reference"].strip()]
    empty_results = {
        it["file_id"]: SemanticWERResult(
            file_id=it["file_id"], chunk_seconds=chunk_seconds, reference_word_count=0,
            affected_word_count=0, semantic_wer=None, status="success",
        )
        for it in items if not it["reference"].strip()
    }
    if not scoreable:
        return [empty_results[it["file_id"]] for it in items]

    pairs_text = "\n\n".join(
        f'[PAIR file_id="{it["file_id"]}"]\n[REFERENCE]\n{it["reference"]}\n\n[HYPOTHESIS]\n{it["hypothesis"]}'
        for it in scoreable
    )
    prompt_content = f"{BATCH_INSTRUCTION}\n\n{pairs_text}"

    def do_call():
        return gemini_client.generate_text(
            client, config.GEMINI_MODEL, contents=[prompt_content],
            response_json_schema=BATCH_JSON_SCHEMA,
        )

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"batch chunk={chunk_seconds}s: attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
    parsed = gemini_client.parse_json_lenient(raw_text)
    by_file_id = {r["file_id"]: r.get("discrepancies", []) for r in parsed.get("results", [])}

    scored_results = {}
    for it in scoreable:
        ref_word_count = len(it["reference"].split())
        discrepancies = [Discrepancy(**d) for d in by_file_id.get(it["file_id"], [])]
        affected = sum(d.word_count_affected for d in discrepancies)
        semantic_wer = affected / ref_word_count if ref_word_count else None
        scored_results[it["file_id"]] = SemanticWERResult(
            file_id=it["file_id"], chunk_seconds=chunk_seconds, reference_word_count=ref_word_count,
            affected_word_count=affected, semantic_wer=semantic_wer, discrepancies=discrepancies, status="success",
        )

    all_results = {**empty_results, **scored_results}
    return [all_results[it["file_id"]] for it in items]


def already_done(source: str, chunk_seconds: float, file_id: str) -> bool:
    p = output_dir(source, chunk_seconds) / f"{file_id}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "success"
    except Exception:
        return False


def write_result(source: str, chunk_seconds: float, result: SemanticWERResult):
    d = output_dir(source, chunk_seconds)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.file_id}.json").write_text(result.model_dump_json(indent=2))


def write_error(source: str, chunk_seconds: float, file_id: str, error: str):
    d = output_dir(source, chunk_seconds)
    d.mkdir(parents=True, exist_ok=True)
    err = SemanticWERResult(file_id=file_id, chunk_seconds=chunk_seconds, reference_word_count=0, affected_word_count=0, semantic_wer=None, status="error", error=error)
    (d / f"{file_id}.json").write_text(err.model_dump_json(indent=2))


def unique_records():
    seen = set()
    for r in dsv2.load_dataset_v2():
        if r.file_id in seen:
            continue
        seen.add(r.file_id)
        yield r


def write_comparison(source: str, chunk_sizes: List[float]):
    import csv

    rows = []
    for chunk_seconds in chunk_sizes:
        d = output_dir(source, chunk_seconds)
        vals = []
        for p in d.glob("*.json"):
            data = json.loads(p.read_text())
            if data.get("status") == "success" and data.get("semantic_wer") is not None:
                vals.append(data["semantic_wer"])
        rows.append({
            "chunk_seconds": chunk_seconds,
            "n_files": len(vals),
            "mean_semantic_wer": sum(vals) / len(vals) if vals else None,
        })
    rows.sort(key=lambda r: r["chunk_seconds"])

    out_path = config.PROJECT_ROOT / "semantic_wer_results" / source / "semantic_wer_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_seconds", "n_files", "mean_semantic_wer"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(SOURCE_DIRS), default="indic_conformer",
                         help="Which ASR pipeline's transcripts to score against the Gemini reference")
    parser.add_argument("--chunk-seconds", type=str, default="2,5,7,10")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5, help="Concurrent Gemini calls (I/O-bound, threading is safe here)")
    parser.add_argument("--batch-size", type=int, default=1,
                         help="Score this many (reference, hypothesis) pairs per Gemini call, e.g. 5-10, instead of one call per file")
    args = parser.parse_args()

    chunk_sizes = [float(x) for x in args.chunk_seconds.split(",")]
    logger = StageLogger(f"semantic_wer_{args.source}")
    client = gemini_client.get_client()

    for chunk_seconds in chunk_sizes:
        records = list(unique_records())
        if args.dry_run:
            records = records[: args.dry_run_limit]
            logger.info(f"DRY RUN source={args.source} chunk_seconds={chunk_seconds}: {len(records)} file(s)")
        else:
            if not args.force:
                before = len(records)
                records = [r for r in records if not already_done(args.source, chunk_seconds, r.file_id)]
                logger.info(f"chunk_seconds={chunk_seconds}: skipping {before - len(records)} already-done file(s)")
            if args.limit:
                records = records[: args.limit]

        n_success, n_error = 0, 0
        n_skipped = 0
        counts_lock = threading.Lock()
        t0 = time.time()

        # Resolve reference/hypothesis up front so skips don't cost a batch slot,
        # then group the scoreable ones into batches of --batch-size.
        scoreable = []
        for record in records:
            reference = load_reference(record)
            hypothesis = load_hypothesis(args.source, chunk_seconds, record.file_id)
            if reference is None or hypothesis is None:
                logger.warn(f"{record.file_id}: missing reference or hypothesis, skipping")
                n_skipped += 1
                continue
            scoreable.append({"file_id": record.file_id, "reference": reference, "hypothesis": hypothesis})

        batch_size = max(1, args.batch_size)
        batches = [scoreable[i:i + batch_size] for i in range(0, len(scoreable), batch_size)]
        logger.info(f"chunk_seconds={chunk_seconds}: {len(scoreable)} file(s) in {len(batches)} batch(es) of up to {batch_size}")

        def process_batch(b_idx, batch_items):
            nonlocal n_success, n_error
            logger.info(f"[chunk={chunk_seconds}s batch {b_idx}/{len(batches)}] {len(batch_items)} file(s): {', '.join(it['file_id'] for it in batch_items)}")
            try:
                if batch_size == 1:
                    results = [compute_semantic_wer(client, batch_items[0]["file_id"], chunk_seconds, batch_items[0]["reference"], batch_items[0]["hypothesis"], logger)]
                else:
                    results = compute_semantic_wer_batch(client, chunk_seconds, batch_items, logger)
                with counts_lock:
                    n_success += len(results)
                for result in results:
                    if args.dry_run:
                        print(result.model_dump_json(indent=2))
                    else:
                        write_result(args.source, chunk_seconds, result)
            except Exception as exc:  # noqa: BLE001
                with counts_lock:
                    n_error += len(batch_items)
                file_ids = ", ".join(it["file_id"] for it in batch_items)
                logger.error(f"batch [{file_ids}]: FAILED — {exc}\n{traceback.format_exc()}")
                if not args.dry_run:
                    for it in batch_items:
                        write_error(args.source, chunk_seconds, it["file_id"], str(exc))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_batch, i, batch) for i, batch in enumerate(batches, 1)]
            for fut in as_completed(futures):
                fut.result()  # re-raise anything that escaped process_batch's own try/except

        logger.info(f"chunk_seconds={chunk_seconds} done in {time.time() - t0:.1f}s — success={n_success} error={n_error} skipped={n_skipped}")

    if not args.dry_run:
        comparison = write_comparison(args.source, chunk_sizes)
        print("\n" + "=" * 80)
        print("SEMANTIC WER COMPARISON")
        print("=" * 80)
        for row in comparison:
            print(row)


if __name__ == "__main__":
    main()
