"""Directly ask Gemini, per file, whether a transcript's difference from
ground truth is a MEANINGFUL/significant difference or not (a holistic
yes/no judgment, not a proxy derived from the semantic-WER discrepancy
word-count), then correlate that against raw WER/CER buckets to find the
threshold past which raw error rate stops being cosmetic noise and starts
reflecting a real difference a reviewer would notice.

This complements wer_threshold_analysis.py (which buckets the ALREADY
existing semantic_wer numbers) by getting a fresh, independent Gemini
judgment per file — useful because a human skimming a transcript makes a
holistic call ("does this feel different"), not a sum-of-discrepancy-word
score.

Samples up to --per-bucket files per WER/CER bucket (uniformly across
bucket, not all ~1500 pairs) to keep this fast and cheap, and asks in
batches of --batch-size.

Usage:
    python3 wer_significance_analysis.py
    python3 wer_significance_analysis.py --sources indic_transcribe_core --per-bucket 10
"""
import argparse
import csv
import json
import random
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import config
import dataset_v2 as dsv2
import gemini_client
from pipeline_logging import StageLogger

SOURCE_DIRS = {
    "indic_conformer": config.PROJECT_ROOT / "indic_conformer_results",
    "indic_transcribe_core": config.PROJECT_ROOT / "indic_transcribe_core_results",
}
CHUNK_SIZES = [2.0, 5.0, 7.0, 10.0]
OUTPUT_DIR = config.PROJECT_ROOT / "analysis_results"
BIN_WIDTH = 0.1
MAX_BIN = 1.5

SIGNIFICANCE_INSTRUCTION = """
You are comparing several (REFERENCE, HYPOTHESIS) transcript pairs of
DIFFERENT audio calls. The REFERENCE is accurate; the HYPOTHESIS comes
from a less accurate speech recognizer of the SAME audio.

For EACH pair, make a HOLISTIC judgment (the way a human reviewer skimming
both transcripts would): is the HYPOTHESIS a MEANINGFULLY/SIGNIFICANTLY
different account of the conversation, or does it read as essentially the
same conversation despite wording/spelling noise? Consider the overall
gist, not just a literal word-count of differences — many small spelling
variants should NOT count as significant, but a few differences that
change who-said-what, numbers, or the topic SHOULD.

Return ONE raw, minified JSON object, one entry per file_id given:
{"results": [{"file_id": "...", "significant": true/false, "reason": "one short phrase"}]}
Your entire response must be only the minified JSON object and nothing else.
""".strip()

SIGNIFICANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                    "significant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["file_id", "significant", "reason"],
            },
        },
    },
    "required": ["results"],
}


def bin_label(value: float) -> str:
    if value >= MAX_BIN:
        return f"{MAX_BIN:.1f}+"
    lo = int(value / BIN_WIDTH) * BIN_WIDTH
    return f"{lo:.1f}-{lo + BIN_WIDTH:.1f}"


def load_ground_truth_text(record: dsv2.FileRecordV2) -> Optional[str]:
    p = record.transcripts_dir / f"{record.file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    return data["transcript"].get("full_text", "")


def load_rows(source: str, source_dir: Path, records_by_id: dict) -> List[dict]:
    rows = []
    for chunk_seconds in CHUNK_SIZES:
        chunk_name = f"chunk_{chunk_seconds:g}s"
        transcripts_dir = source_dir / chunk_name / "transcripts"
        if not transcripts_dir.is_dir():
            continue
        for p in transcripts_dir.glob("*.json"):
            data = json.loads(p.read_text())
            if data.get("status") != "success" or not data.get("metrics"):
                continue
            wer, cer = data["metrics"].get("wer"), data["metrics"].get("cer")
            if not wer or not cer or wer["wer"] is None or cer["cer"] is None:
                continue
            record = records_by_id.get(data["file_id"])
            if record is None:
                continue
            reference = load_ground_truth_text(record)
            if not reference:
                continue
            rows.append({
                "file_id": data["file_id"], "chunk_seconds": chunk_seconds,
                "wer": wer["wer"], "cer": cer["cer"],
                "reference": reference, "hypothesis": data.get("full_text", ""),
            })
    return rows


def sample_per_bucket(rows: List[dict], key: str, per_bucket: int, seed: int = 0) -> List[dict]:
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[bin_label(r[key])].append(r)
    rng = random.Random(seed)
    sampled = []
    for bucket, bucket_rows in by_bucket.items():
        rng.shuffle(bucket_rows)
        sampled.extend(bucket_rows[:per_bucket])
    return sampled


def judge_batch(client, batch: List[dict], logger: StageLogger) -> dict:
    file_blocks = [
        f'[FILE file_id="{r["file_id"]}"]\n[REFERENCE]\n{r["reference"]}\n\n[HYPOTHESIS]\n{r["hypothesis"]}'
        for r in batch
    ]
    prompt_content = f"{SIGNIFICANCE_INSTRUCTION}\n\n" + "\n\n".join(file_blocks)

    def do_call():
        return gemini_client.generate_text(client, config.GEMINI_MODEL, contents=[prompt_content], response_json_schema=SIGNIFICANCE_SCHEMA)

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"batch attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
    parsed = gemini_client.parse_json_lenient(raw_text)
    return {r["file_id"]: r for r in parsed.get("results", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, default=None)
    parser.add_argument("--per-bucket", type=int, default=15, help="Max files sampled per WER bucket (and separately per CER bucket)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else list(SOURCE_DIRS)
    logger = StageLogger("wer_significance_analysis")
    client = gemini_client.get_client()
    records_by_id = {r.file_id: r for r in dsv2.load_dataset_v2()}

    all_bucket_rows = []

    for source in sources:
        all_rows = load_rows(source, SOURCE_DIRS[source], records_by_id)
        wer_sample = sample_per_bucket(all_rows, "wer", args.per_bucket)
        cer_sample = sample_per_bucket(all_rows, "cer", args.per_bucket)
        combined = {r["file_id"] + str(r["chunk_seconds"]): r for r in wer_sample + cer_sample}
        to_judge = list(combined.values())
        logger.info(f"[{source}] judging {len(to_judge)} sampled (file, chunk) pairs across WER/CER buckets...")

        batches = [to_judge[i:i + args.batch_size] for i in range(0, len(to_judge), args.batch_size)]
        judgments = {}
        lock = threading.Lock()

        def process_batch(b_idx, batch):
            logger.info(f"[{source} batch {b_idx}/{len(batches)}] {len(batch)} file(s)")
            result = judge_batch(client, batch, logger)
            with lock:
                judgments.update(result)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_batch, i, b) for i, b in enumerate(batches, 1)]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[{source}] batch failed: {exc}\n{traceback.format_exc()}")

        for key_name, key_metric in (("wer", "wer"), ("cer", "cer")):
            by_bucket = defaultdict(list)
            for r in to_judge:
                j = judgments.get(r["file_id"])
                if j is None:
                    continue
                by_bucket[bin_label(r[key_metric])].append(j["significant"])
            for bucket, vals in sorted(by_bucket.items(), key=lambda kv: float(kv[0].split("-")[0].rstrip("+"))):
                all_bucket_rows.append({
                    "source": source, "metric": key_name, "bucket": bucket,
                    "n_judged": len(vals), "pct_significant": sum(vals) / len(vals),
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "wer_cer_significance_by_bucket.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "metric", "bucket", "n_judged", "pct_significant"])
        writer.writeheader()
        writer.writerows(all_bucket_rows)

    print("\n" + "=" * 90)
    print("GEMINI-JUDGED SIGNIFICANCE BY RAW WER/CER BUCKET")
    print("=" * 90)
    for row in all_bucket_rows:
        print(f"{row['source']:22} {row['metric']:4} {row['bucket']:>8}  n={row['n_judged']:3d}  pct_significant={row['pct_significant']:.2f}")
    print(f"\nSaved to {OUTPUT_DIR}/wer_cer_significance_by_bucket.csv")


if __name__ == "__main__":
    main()
