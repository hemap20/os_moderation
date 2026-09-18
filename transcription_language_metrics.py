"""Per-language micro-averaged WER/CER (and semantic WER) for the ASR
transcription pipelines (Indic-Conformer, Indic-Transcribe-Core), across
every chunk size already run. Reads straight from each pipeline's
{results_dir}/chunk_{X}s/transcripts/*.json (which already carries the
file's language and per-file wer/cer ref counts) plus
semantic_wer_results/{source}/chunk_{X}s/*.json for the semantic scores.

Micro-averaging (total errors / total reference units, not a naive mean of
per-file ratios) is used for WER/CER, same reasoning as
indic_transcribe_core_pipeline.py's own report — a few near-silent outlier
files with tiny reference word counts would otherwise dominate a per-file
mean out of proportion to their actual content.

Usage:
    python3 transcription_language_metrics.py
    python3 transcription_language_metrics.py --sources indic_conformer
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import config

TRANSCRIPTION_DIRS = {
    "indic_conformer": config.PROJECT_ROOT / "indic_conformer_results",
    "indic_transcribe_core": config.PROJECT_ROOT / "indic_transcribe_core_results",
}

CHUNK_SIZES = [2.0, 5.0, 7.0, 10.0]

OUTPUT_DIR = config.PROJECT_ROOT / "analysis_results"


def chunk_dir_name(chunk_seconds: float) -> str:
    return f"chunk_{chunk_seconds:g}s"


def micro_average(rows, ratio_key, count_key) -> Optional[float]:
    total_err = sum(r[ratio_key] * r[count_key] for r in rows if r[count_key])
    total_count = sum(r[count_key] for r in rows if r[count_key])
    return total_err / total_count if total_count else None


def load_wer_cer_rows(source_dir: Path, chunk_seconds: float) -> list:
    transcripts_dir = source_dir / chunk_dir_name(chunk_seconds) / "transcripts"
    rows = []
    if not transcripts_dir.is_dir():
        return rows
    for p in transcripts_dir.glob("*.json"):
        data = json.loads(p.read_text())
        if data.get("status") != "success":
            continue
        metrics = data.get("metrics") or {}
        wer, cer = metrics.get("wer"), metrics.get("cer")
        if not wer or not cer:
            continue
        rows.append({
            "language": data.get("language", "unknown"),
            "wer": wer["wer"], "wer_ref_word_count": wer["ref_word_count"],
            "cer": cer["cer"], "cer_ref_char_count": cer["ref_char_count"],
        })
    return rows


def load_semantic_wer_rows(source: str, chunk_seconds: float, language_by_file: dict) -> list:
    d = config.PROJECT_ROOT / "semantic_wer_results" / source / chunk_dir_name(chunk_seconds)
    rows = []
    if not d.is_dir():
        return rows
    for p in d.glob("*.json"):
        data = json.loads(p.read_text())
        if data.get("status") != "success" or data.get("semantic_wer") is None:
            continue
        language = language_by_file.get(data["file_id"])
        if language is None:
            continue
        rows.append({
            "language": language,
            "semantic_wer": data["semantic_wer"],
            "reference_word_count": data["reference_word_count"],
        })
    return rows


def build_language_by_file(source_dir: Path, chunk_seconds: float) -> dict:
    transcripts_dir = source_dir / chunk_dir_name(chunk_seconds) / "transcripts"
    out = {}
    if not transcripts_dir.is_dir():
        return out
    for p in transcripts_dir.glob("*.json"):
        data = json.loads(p.read_text())
        out[data["file_id"]] = data.get("language", "unknown")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, default=None, help="Comma-separated subset; default = all")
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else list(TRANSCRIPTION_DIRS)

    wer_cer_rows = []
    semantic_rows = []

    for source in sources:
        source_dir = TRANSCRIPTION_DIRS[source]
        for chunk_seconds in CHUNK_SIZES:
            rows = load_wer_cer_rows(source_dir, chunk_seconds)
            if not rows:
                continue
            by_lang = defaultdict(list)
            for r in rows:
                by_lang[r["language"]].append(r)
            for language, lang_rows in sorted(by_lang.items()):
                wer_cer_rows.append({
                    "source": source, "chunk_seconds": chunk_seconds, "language": language,
                    "n_files": len(lang_rows),
                    "micro_wer": micro_average(lang_rows, "wer", "wer_ref_word_count"),
                    "micro_cer": micro_average(lang_rows, "cer", "cer_ref_char_count"),
                })

            language_by_file = build_language_by_file(source_dir, chunk_seconds)
            sem_rows = load_semantic_wer_rows(source, chunk_seconds, language_by_file)
            if not sem_rows:
                continue
            by_lang_sem = defaultdict(list)
            for r in sem_rows:
                by_lang_sem[r["language"]].append(r)
            for language, lang_rows in sorted(by_lang_sem.items()):
                semantic_rows.append({
                    "source": source, "chunk_seconds": chunk_seconds, "language": language,
                    "n_files": len(lang_rows),
                    "micro_semantic_wer": micro_average(lang_rows, "semantic_wer", "reference_word_count"),
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "transcription_wer_cer_by_language.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "chunk_seconds", "language", "n_files", "micro_wer", "micro_cer"])
        writer.writeheader()
        writer.writerows(wer_cer_rows)
    with open(OUTPUT_DIR / "transcription_semantic_wer_by_language.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "chunk_seconds", "language", "n_files", "micro_semantic_wer"])
        writer.writeheader()
        writer.writerows(semantic_rows)

    print("=" * 90)
    print("WER / CER BY LANGUAGE (micro-averaged)")
    print("=" * 90)
    for row in wer_cer_rows:
        print(row)
    print()
    print("=" * 90)
    print("SEMANTIC WER BY LANGUAGE (micro-averaged)")
    print("=" * 90)
    for row in semantic_rows:
        print(row)

    print(f"\nSaved to {OUTPUT_DIR}/transcription_wer_cer_by_language.csv and transcription_semantic_wer_by_language.csv")


if __name__ == "__main__":
    main()
