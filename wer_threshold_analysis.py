"""How much does raw WER/CER have to be before it reflects an ACTUAL
meaning-changing difference (semantic WER), rather than cosmetic noise
(spelling/phonetic variants, filler words)?

Joins each file's raw WER/CER (from {source}_results/chunk_{X}s/transcripts/)
with its semantic WER (from semantic_wer_results/{source}/chunk_{X}s/) by
file_id + chunk_seconds, buckets by raw WER (and separately by raw CER)
into 0.1-wide bins, and reports the mean semantic WER per bin — the bin
where mean semantic WER starts climbing sharply is the threshold past
which a high raw WER stops being just noise.

Usage:
    python3 wer_threshold_analysis.py
    python3 wer_threshold_analysis.py --sources indic_transcribe_core
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import config

SOURCE_DIRS = {
    "indic_conformer": config.PROJECT_ROOT / "indic_conformer_results",
    "indic_transcribe_core": config.PROJECT_ROOT / "indic_transcribe_core_results",
}
CHUNK_SIZES = [2.0, 5.0, 7.0, 10.0]
OUTPUT_DIR = config.PROJECT_ROOT / "analysis_results"

BIN_WIDTH = 0.1
MAX_BIN = 1.5  # anything >= this is lumped into a final "1.5+" bin


def bin_label(value: float) -> str:
    if value >= MAX_BIN:
        return f"{MAX_BIN:.1f}+"
    lo = int(value / BIN_WIDTH) * BIN_WIDTH
    return f"{lo:.1f}-{lo + BIN_WIDTH:.1f}"


def load_joined_rows(source: str, source_dir: Path) -> List[dict]:
    rows = []
    for chunk_seconds in CHUNK_SIZES:
        chunk_name = f"chunk_{chunk_seconds:g}s"
        transcripts_dir = source_dir / chunk_name / "transcripts"
        sem_dir = config.PROJECT_ROOT / "semantic_wer_results" / source / chunk_name
        if not transcripts_dir.is_dir() or not sem_dir.is_dir():
            continue
        for p in transcripts_dir.glob("*.json"):
            data = json.loads(p.read_text())
            if data.get("status") != "success" or not data.get("metrics"):
                continue
            wer = data["metrics"].get("wer")
            cer = data["metrics"].get("cer")
            if not wer or not cer:
                continue
            sem_path = sem_dir / p.name
            if not sem_path.exists():
                continue
            sem_data = json.loads(sem_path.read_text())
            if sem_data.get("status") != "success" or sem_data.get("semantic_wer") is None:
                continue
            if wer["wer"] is None or cer["cer"] is None:
                continue
            rows.append({
                "file_id": data["file_id"],
                "language": data.get("language", "unknown"),
                "chunk_seconds": chunk_seconds,
                "wer": wer["wer"],
                "cer": cer["cer"],
                "semantic_wer": sem_data["semantic_wer"],
            })
    return rows


def bucket_by(rows: List[dict], key: str) -> List[dict]:
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[bin_label(r[key])].append(r["semantic_wer"])

    def sort_key(label: str):
        return float(label.split("-")[0].rstrip("+"))

    table = []
    for label in sorted(by_bin, key=sort_key):
        vals = by_bin[label]
        table.append({
            "bucket": label,
            "n_files": len(vals),
            "mean_semantic_wer": round(sum(vals) / len(vals), 2),
            "min_semantic_wer": round(min(vals), 2),
            "max_semantic_wer": round(max(vals), 2),
        })
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, default=None)
    args = parser.parse_args()
    sources = args.sources.split(",") if args.sources else list(SOURCE_DIRS)

    all_wer_rows, all_cer_rows = [], []

    for source in sources:
        rows = load_joined_rows(source, SOURCE_DIRS[source])
        print(f"\n{'=' * 80}\n{source}: {len(rows)} (file, chunk) pairs with both raw and semantic WER\n{'=' * 80}")

        wer_table = bucket_by(rows, "wer")
        print("\nBy raw WER bucket -> mean semantic WER:")
        for row in wer_table:
            print(f"  WER {row['bucket']:>8}  n={row['n_files']:4d}  mean_semantic_wer={row['mean_semantic_wer']:.2f}  (range {row['min_semantic_wer']:.2f}-{row['max_semantic_wer']:.2f})")
        for row in wer_table:
            all_wer_rows.append({"source": source, **row})

        cer_table = bucket_by(rows, "cer")
        print("\nBy raw CER bucket -> mean semantic WER:")
        for row in cer_table:
            print(f"  CER {row['bucket']:>8}  n={row['n_files']:4d}  mean_semantic_wer={row['mean_semantic_wer']:.2f}  (range {row['min_semantic_wer']:.2f}-{row['max_semantic_wer']:.2f})")
        for row in cer_table:
            all_cer_rows.append({"source": source, **row})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "wer_vs_semantic_wer_buckets.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "bucket", "n_files", "mean_semantic_wer", "min_semantic_wer", "max_semantic_wer"])
        writer.writeheader()
        writer.writerows(all_wer_rows)
    with open(OUTPUT_DIR / "cer_vs_semantic_wer_buckets.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "bucket", "n_files", "mean_semantic_wer", "min_semantic_wer", "max_semantic_wer"])
        writer.writeheader()
        writer.writerows(all_cer_rows)
    print(f"\nSaved to {OUTPUT_DIR}/wer_vs_semantic_wer_buckets.csv and cer_vs_semantic_wer_buckets.csv")


if __name__ == "__main__":
    main()
