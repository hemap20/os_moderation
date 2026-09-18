"""Shared reporting for the ASR transcription pipelines (Indic-Conformer,
Indic-Transcribe-Core): an overall per-chunk-size comparison (micro-averaged
WER/CER, mean boundary-corruption rate / policy-term recall / real-time
factor) and a per-language pivot table — one row per language, one column
per metric x chunk size (WER, CER, semantic WER, boundary corruption,
policy-term recall, real-time factor).

Both reports are read straight from each file's own on-disk
{output_dir}/chunk_{X}s/transcripts/*.json rather than the current run's
in-memory results list, so a resumed/partial run's report reflects every
file completed so far (across all runs), not just this run's subset — this
is the same fix already applied to indic_transcribe_core_pipeline.py's own
report, now shared by both pipelines and extended with the per-language
breakdown.

Micro-averaging (total errors / total reference units, not a plain mean of
each file's own ratio) is used for WER, CER, and semantic WER, since a
per-file mean gets skewed by near-silent-call outliers with tiny reference
word/char counts. Boundary-corruption rate and policy-term recall are also
micro-averaged (weighted by each file's own boundary/excerpt count).
Real-time factor has no natural per-file weight, so it's a plain mean.

Semantic WER is included in the per-language pivot ONLY if
semantic_wer.py has already been run for that source/chunk size (reads
semantic_wer_results/{source}/chunk_{X}s/*.json if present) — it's left
blank otherwise, since this module has no Gemini dependency of its own.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import config


def chunk_dir_name(chunk_seconds: float) -> str:
    return f"chunk_{chunk_seconds:g}s"


def micro_average(rows: List[dict], ratio_key: str, count_key: str) -> Optional[float]:
    total_err = sum(r[ratio_key] * r[count_key] for r in rows if r.get(count_key))
    total_count = sum(r[count_key] for r in rows if r.get(count_key))
    return total_err / total_count if total_count else None


def mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load_file_rows(output_root: Path, chunk_seconds: float) -> List[dict]:
    """One row per successfully-transcribed file for this chunk size, with
    every metric plus its natural weighting count (ref word/char count for
    WER/CER, n_boundaries for boundary corruption, n_excerpts for
    policy-term recall)."""
    transcripts_dir = output_root / chunk_dir_name(chunk_seconds) / "transcripts"
    rows = []
    if not transcripts_dir.is_dir():
        return rows
    for p in transcripts_dir.glob("*.json"):
        data = json.loads(p.read_text())
        if data.get("status") != "success":
            continue
        m = data.get("metrics")
        if not m:
            continue
        wer, cer = m.get("wer"), m.get("cer")
        boundary = m.get("boundary_corruption") or {}
        policy = m.get("policy_term_recall") or {}
        rows.append({
            "file_id": data["file_id"],
            "language": data.get("language", "unknown"),
            "wer": wer["wer"] if wer else None,
            "wer_ref_word_count": wer["ref_word_count"] if wer else 0,
            "cer": cer["cer"] if cer else None,
            "cer_ref_char_count": cer["ref_char_count"] if cer else 0,
            "boundary_corruption_rate": boundary.get("rate"),
            "n_boundaries": boundary.get("n_boundaries", 0),
            "n_corrupted": boundary.get("n_corrupted", 0),
            "policy_term_recall": policy.get("recall"),
            "n_excerpts": policy.get("n_excerpts", 0),
            "n_preserved": policy.get("n_preserved", 0),
            "real_time_factor": m.get("real_time_factor"),
        })
    return rows


def load_semantic_wer_rows(source: str, chunk_seconds: float, language_by_file: Dict[str, str]) -> List[dict]:
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


def write_overall_comparison(output_root: Path, chunk_sizes: List[float]) -> List[dict]:
    comparison = []
    for chunk_seconds in chunk_sizes:
        rows = load_file_rows(output_root, chunk_seconds)
        if not rows:
            continue
        comparison.append({
            "chunk_seconds": chunk_seconds,
            "n_files": len(rows),
            "micro_wer": micro_average(rows, "wer", "wer_ref_word_count"),
            "micro_cer": micro_average(rows, "cer", "cer_ref_char_count"),
            "mean_boundary_corruption_rate": mean([r["boundary_corruption_rate"] for r in rows]),
            "mean_policy_term_recall": mean([r["policy_term_recall"] for r in rows]),
            "mean_real_time_factor": mean([r["real_time_factor"] for r in rows]),
        })
    comparison.sort(key=lambda r: r["chunk_seconds"])

    out_path = output_root / "chunk_size_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chunk_seconds", "n_files", "micro_wer", "micro_cer",
            "mean_boundary_corruption_rate", "mean_policy_term_recall", "mean_real_time_factor",
        ])
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


PIVOT_METRICS = ["wer", "cer", "swer", "boundary_corruption", "policy_recall", "rtf"]


def write_language_pivot(source: str, output_root: Path, chunk_sizes: List[float]) -> Path:
    """One row per language; columns are {metric}_{chunk}s for wer, cer,
    swer, boundary_corruption, policy_recall, rtf. Saved as
    {output_root}/language_metrics_pivot.csv."""
    per_lang: Dict[str, Dict[tuple, Optional[float]]] = defaultdict(dict)

    for chunk_seconds in chunk_sizes:
        rows = load_file_rows(output_root, chunk_seconds)
        if not rows:
            continue
        by_lang = defaultdict(list)
        for r in rows:
            by_lang[r["language"]].append(r)

        for language, lang_rows in by_lang.items():
            per_lang[language][("wer", chunk_seconds)] = micro_average(lang_rows, "wer", "wer_ref_word_count")
            per_lang[language][("cer", chunk_seconds)] = micro_average(lang_rows, "cer", "cer_ref_char_count")
            n_boundaries_total = sum(r["n_boundaries"] for r in lang_rows)
            per_lang[language][("boundary_corruption", chunk_seconds)] = (
                sum(r["n_corrupted"] for r in lang_rows) / n_boundaries_total if n_boundaries_total else None
            )
            n_excerpts_total = sum(r["n_excerpts"] for r in lang_rows)
            per_lang[language][("policy_recall", chunk_seconds)] = (
                sum(r["n_preserved"] for r in lang_rows) / n_excerpts_total if n_excerpts_total else None
            )
            per_lang[language][("rtf", chunk_seconds)] = mean([r["real_time_factor"] for r in lang_rows])

        language_by_file = {r["file_id"]: r["language"] for r in rows}
        sem_rows = load_semantic_wer_rows(source, chunk_seconds, language_by_file)
        sem_by_lang = defaultdict(list)
        for r in sem_rows:
            sem_by_lang[r["language"]].append(r)
        for language, lang_rows in sem_by_lang.items():
            per_lang[language][("swer", chunk_seconds)] = micro_average(lang_rows, "semantic_wer", "reference_word_count")

    cols = [f"{m}_{c:g}s" for m in PIVOT_METRICS for c in chunk_sizes]
    out_path = output_root / "language_metrics_pivot.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language"] + cols)
        for language in sorted(per_lang):
            vals = per_lang[language]
            writer.writerow([language] + [
                f"{vals[(m, c)]:.4f}" if vals.get((m, c)) is not None else ""
                for m in PIVOT_METRICS for c in chunk_sizes
            ])
    return out_path


def write_all_reports(source: str, output_root: Path, chunk_sizes: List[float]) -> None:
    """Convenience entry point for a pipeline's main() to call once at the
    end of a run: writes both {output_root}/chunk_size_comparison.csv and
    {output_root}/language_metrics_pivot.csv."""
    comparison = write_overall_comparison(output_root, chunk_sizes)
    pivot_path = write_language_pivot(source, output_root, chunk_sizes)

    print("\n" + "=" * 80)
    print(f"CHUNK SIZE COMPARISON ({source})")
    print("=" * 80)
    for row in comparison:
        print(row)
    print(f"\nPer-language metrics (WER/CER/semantic-WER/boundary-corruption/policy-recall/RTF")
    print(f"x chunk size) saved to: {pivot_path}")
