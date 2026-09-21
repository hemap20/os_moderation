"""Score every model's results against ground truth (Stage 1+2's
determination, produced by stage12_v2.py) and save confusion-matrix-derived
metrics, both overall and split by language.

Ground truth per file = classification.ground_truth_flags in
{Dostt}/{lang}_{cat}/transcripts/{file_id}.json or {Dostt}/{lang}_fn/transcripts/{file_id}.json
(the Gemini-confirmed violation list, NOT the raw folder bucket name).

Matching rule (per user's explicit spec):
  FLAG level: a ground-truth flag and a model flag match if they share the
  SAME category AND are semantically the same underlying quote/incident
  (judged by translation content). One ground-truth flag missed by the
  model = 1 FN. One model flag with no matching ground-truth flag = 1 FP.
  Each matched pair = 1 TP. (No flag-level TN — "nothing happened" isn't a
  countable unit.)

  FILE level: ground truth positive = file has >=1 ground-truth flag.
  Model positive = model raised >=1 flag. File is TP if ground truth is
  positive AND at least one of the model's flags matched (flag-level TP)
  ANY ground-truth flag on that file — i.e. the model caught the category
  correctly at least once, regardless of extra wrong flags. FN if ground
  truth positive but no matched flag. FP if ground truth negative but
  model raised >=1 flag. TN if ground truth negative and model raised none.

Matching is done ENTIRELY by Gemini, no Python shortcut: every file's
full ground-truth flag list and model flag list (category + translation)
is sent to Gemini, batched 5 files per call, and Gemini alone decides
which specific flags correspond to the same incident (same category AND
same semantic content) — including the trivial one-vs-one case, which
used to be auto-matched locally but is now judged by the model like
everything else. Python only turns Gemini's returned (gt_index,
model_index) pairs into TP/FP/FN counts — no arithmetic judgment calls
are made by Gemini, only the matching decision.

Confidence-bucket precision (flag level, discrete bins, precision only —
per your choice): each model flag is binned by its confidence — once using
the model's self-reported model_confidence, once using
logprob_derived_confidence — into [0.0,0.2) ... [0.8,1.0], and precision =
matched / (matched + unmatched) is reported per bin. Recall/specificity/
accuracy aren't bucketable this way (model emits no confidence for the
files it call clean), so those are reported once, unbucketed, at file
level.

Token entropy (excerpt_token_entropy.mean) is reported descriptively:
mean entropy for TP vs FP flags, and mean entropy per confidence bucket —
a high mean entropy among FP/low-confidence flags would suggest the model
is quoting unclear audio it isn't sure about, which is a candidate signal
for filtering unreliable flags independent of its confidence score.

Usage:
    python3 analyze_results.py
    python3 analyze_results.py --models e2b_nothinking,gemini-3.5-flash-lite
    python3 analyze_results.py --dry-run --dry-run-limit 5
"""
import argparse
import csv
import json
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
import dataset_v2 as dsv2
import gemini_client
from pipeline_logging import StageLogger

MODEL_DIRS: Dict[str, Path] = {
    "e2b_thinking": config.PROJECT_ROOT / "gemma_results" / "e2b_thinking",
    "e2b_nothinking": config.PROJECT_ROOT / "gemma_results" / "e2b_nothinking",
    "e4b_thinking": config.PROJECT_ROOT / "gemma_results" / "e4b_thinking",
    "e4b_nothinking": config.PROJECT_ROOT / "gemma_results" / "e4b_nothinking",
    "12b_nothinking": config.PROJECT_ROOT / "gemma_results" / "12b_nothinking",
    "e2b_conformer5s_text_nothinking": config.PROJECT_ROOT / "gemma_results" / "e2b_conformer5s_text_nothinking",
    "e4b_conformer5s_text_nothinking": config.PROJECT_ROOT / "gemma_results" / "e4b_conformer5s_text_nothinking",
    "gemini-3.5-flash-lite": config.PROJECT_ROOT / "gemini_results" / "gemini-3.5-flash-lite",
}

OUTPUT_DIR = config.PROJECT_ROOT / "analysis_results"

CONFIDENCE_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{min(hi, 1.0):.1f}"


BATCH_MATCH_INSTRUCTION = """
You are matching ground-truth policy-violation flags against a candidate
model's flags, across SEVERAL different audio calls at once. For each
call (identified by file_id), you are given a GT list and a MODEL list,
each item being {category, translation} describing a possible violation
in that call.

For each file, decide which GT item and MODEL item refer to the SAME
underlying incident: they must share the SAME category AND describe the
same specific quote/moment (semantically — translation wording may
differ, but it must be the same thing being said, not just the same
category in general). Each GT item may match AT MOST one MODEL item and
vice versa. Do not force a match — a genuine miss (a GT flag nothing in
MODEL corresponds to) or a genuine extra/wrong flag (a MODEL flag nothing
in GT corresponds to) are valid, expected outcomes and should be left
unmatched.

Return ONE raw, minified JSON object:
{"results": [{"file_id": "...", "pairs": [[gt_index, model_index], ...]}]}
Indices are 0-based into that file's own GT/MODEL lists as given below.
Include EVERY file_id given, even if its pairs list ends up empty. Your
entire response must be only the minified JSON object and nothing else.
""".strip()

BATCH_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                    "pairs": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    },
                },
                "required": ["file_id", "pairs"],
            },
        },
    },
    "required": ["results"],
}


def load_ground_truth(record: dsv2.FileRecordV2) -> Optional[List[dict]]:
    p = record.transcripts_dir / f"{record.file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    cls = data.get("classification", {})
    if cls.get("status") != "success":
        return None
    return [
        {"category": f["ground_truth_category"], "translation": f["ground_truth_translation"]}
        for f in cls.get("ground_truth_flags", [])
    ]


def load_model_flags(model_dir: Path, file_id: str) -> Optional[List[dict]]:
    p = model_dir / "results" / f"{file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    out = []
    for f in data.get("flags", []):
        entropy = f.get("excerpt_token_entropy") or {}
        out.append({
            "category": f.get("model_category", ""),
            "translation": f.get("model_translation", ""),
            "model_confidence": f.get("model_confidence"),
            "logprob_derived_confidence": f.get("logprob_derived_confidence"),
            "entropy_mean": entropy.get("mean"),
        })
    return out


def match_files_batch(client, batch_items: List[dict], logger: StageLogger) -> Dict[str, List[Tuple[int, int]]]:
    """batch_items: list of {file_id, gt_flags, model_flags}. Returns file_id -> matched (gt_idx, model_idx) pairs.
    Files with no flags on either side are skipped locally (nothing to match)."""
    scoreable = [it for it in batch_items if it["gt_flags"] or it["model_flags"]]
    result: Dict[str, List[Tuple[int, int]]] = {it["file_id"]: [] for it in batch_items}
    if not scoreable:
        return result

    file_blocks = []
    for it in scoreable:
        gt_lines = "\n".join(f'  GT[{i}] category={f["category"]} translation="{f["translation"]}"' for i, f in enumerate(it["gt_flags"]))
        model_lines = "\n".join(f'  MODEL[{i}] category={f["category"]} translation="{f["translation"]}"' for i, f in enumerate(it["model_flags"]))
        file_blocks.append(f'[FILE file_id="{it["file_id"]}"]\nGT:\n{gt_lines or "  (none)"}\nMODEL:\n{model_lines or "  (none)"}')
    prompt_content = f"{BATCH_MATCH_INSTRUCTION}\n\n" + "\n\n".join(file_blocks)

    def do_call():
        return gemini_client.generate_text(client, config.GEMINI_MODEL, contents=[prompt_content], response_json_schema=BATCH_MATCH_SCHEMA)

    def on_retry(attempt, max_retries, delay, exc):
        file_ids = ", ".join(it["file_id"] for it in scoreable)
        logger.warn(f"batch [{file_ids}]: match attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
    parsed = gemini_client.parse_json_lenient(raw_text)

    by_file = {it["file_id"]: it for it in scoreable}
    for entry in parsed.get("results", []):
        file_id = entry.get("file_id")
        item = by_file.get(file_id)
        if item is None:
            continue
        pairs = []
        used_gt, used_model = set(), set()
        for a_i, b_i in entry.get("pairs", []):
            if a_i in used_gt or b_i in used_model:
                continue
            if not (0 <= a_i < len(item["gt_flags"])) or not (0 <= b_i < len(item["model_flags"])):
                continue
            used_gt.add(a_i)
            used_model.add(b_i)
            pairs.append((a_i, b_i))
        result[file_id] = pairs
    return result


def unique_records():
    seen = set()
    for r in dsv2.load_dataset_v2():
        if r.file_id in seen:
            continue
        seen.add(r.file_id)
        yield r


def build_score_input(record: dsv2.FileRecordV2, model_dir: Path) -> Optional[dict]:
    gt_flags = load_ground_truth(record)
    model_flags = load_model_flags(model_dir, record.file_id)
    if gt_flags is None or model_flags is None:
        return None
    return {"file_id": record.file_id, "language": record.language, "gt_flags": gt_flags, "model_flags": model_flags}


def score_file(item: dict, matched_pairs: List[Tuple[int, int]]) -> dict:
    gt_flags, model_flags, record_language = item["gt_flags"], item["model_flags"], item["language"]
    matched_gt = {p[0] for p in matched_pairs}
    matched_model = {p[1] for p in matched_pairs}

    flag_tp = len(matched_pairs)
    flag_fn = len(gt_flags) - len(matched_gt)
    flag_fp = len(model_flags) - len(matched_model)

    gt_positive = len(gt_flags) > 0
    model_raised_any = len(model_flags) > 0
    if gt_positive and flag_tp > 0:
        file_bucket = "TP"
    elif gt_positive and flag_tp == 0:
        file_bucket = "FN"
    elif not gt_positive and model_raised_any:
        file_bucket = "FP"
    else:
        file_bucket = "TN"

    flags_detail = []
    for i, f in enumerate(model_flags):
        flags_detail.append({**f, "matched": i in matched_model})

    return {
        "file_id": item["file_id"],
        "language": record_language,
        "gt_flag_count": len(gt_flags),
        "model_flag_count": len(model_flags),
        "flag_tp": flag_tp,
        "flag_fn": flag_fn,
        "flag_fp": flag_fp,
        "file_bucket": file_bucket,
        "model_flags": flags_detail,
    }


def r2(x: Optional[float]) -> Optional[float]:
    return round(x, 2) if x is not None else None


def confusion_metrics(tp: int, fp: int, fn: int, tn: Optional[int] = None) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": r2(precision), "recall": r2(recall)}
    if tn is not None:
        out["tn"] = tn
        out["specificity"] = r2(tn / (tn + fp) if (tn + fp) else None)
        total = tp + fp + fn + tn
        out["accuracy"] = r2((tp + tn) / total if total else None)
    return out


def aggregate_file_level(rows: List[dict]) -> dict:
    tp = sum(1 for r in rows if r["file_bucket"] == "TP")
    fp = sum(1 for r in rows if r["file_bucket"] == "FP")
    fn = sum(1 for r in rows if r["file_bucket"] == "FN")
    tn = sum(1 for r in rows if r["file_bucket"] == "TN")
    return {"n_files": len(rows), **confusion_metrics(tp, fp, fn, tn)}


def aggregate_flag_level(rows: List[dict]) -> dict:
    """Per your spec: TP/FP/FN are counted per flag (matched pair = TP, unmatched
    ground-truth flag = FN, unmatched model flag = FP). TN isn't a per-flag
    thing (a flag has to be raised to exist) — it's counted once per FILE,
    when a file has zero ground-truth flags AND zero model flags."""
    tp = sum(r["flag_tp"] for r in rows)
    fp = sum(r["flag_fp"] for r in rows)
    fn = sum(r["flag_fn"] for r in rows)
    tn = sum(1 for r in rows if r["gt_flag_count"] == 0 and r["model_flag_count"] == 0)
    return confusion_metrics(tp, fp, fn, tn)


def confidence_bucket_table(rows: List[dict], confidence_key: str) -> List[dict]:
    all_flags = [f for r in rows for f in r["model_flags"] if f.get(confidence_key) is not None]
    table = []
    for lo, hi in CONFIDENCE_BINS:
        in_bin = [f for f in all_flags if lo <= f[confidence_key] < hi]
        tp = sum(1 for f in in_bin if f["matched"])
        fp = len(in_bin) - tp
        entropies = [f["entropy_mean"] for f in in_bin if f.get("entropy_mean") is not None]
        table.append({
            "bucket": bucket_label(lo, hi),
            "n_flags": len(in_bin),
            "tp": tp,
            "fp": fp,
            "precision": r2(tp / len(in_bin) if in_bin else None),
            "mean_entropy": r2(sum(entropies) / len(entropies) if entropies else None),
        })
    return table


def entropy_by_correctness(rows: List[dict]) -> dict:
    """Mean excerpt-token entropy split by whether the flag was matched
    (TP) or not (FP) — higher entropy on FP flags would mean the model was
    quoting text it wasn't actually sure about."""
    all_flags = [f for r in rows for f in r["model_flags"]]
    matched = [f["entropy_mean"] for f in all_flags if f["matched"] and f.get("entropy_mean") is not None]
    unmatched = [f["entropy_mean"] for f in all_flags if not f["matched"] and f.get("entropy_mean") is not None]
    return {
        "mean_entropy_tp_flags": r2(sum(matched) / len(matched) if matched else None),
        "n_tp_flags_with_entropy": len(matched),
        "mean_entropy_fp_flags": r2(sum(unmatched) / len(unmatched) if unmatched else None),
        "n_fp_flags_with_entropy": len(unmatched),
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_and_write_csv(path: Path, new_rows: List[dict], fieldnames: List[str], models_being_replaced: List[str]):
    """Re-running a subset of models must not wipe out other models' rows already
    on disk from a previous run — keep existing rows for models NOT in this run,
    replace rows for models that ARE in this run."""
    existing = []
    if path.exists():
        with open(path, newline="") as f:
            existing = list(csv.DictReader(f))
    kept = [r for r in existing if r.get("model") not in models_being_replaced]
    write_csv(path, kept + new_rows, fieldnames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None, help="Comma-separated subset of model keys; default = all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true", help="Recompute per-file scoring even if a cached per-file result exists")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=5, help="Files' worth of GT+model flags sent to Gemini per matching call")
    args = parser.parse_args()

    model_keys = args.models.split(",") if args.models else list(MODEL_DIRS)
    logger = StageLogger("analyze_results")
    client = gemini_client.get_client()

    records = list(unique_records())
    if args.dry_run:
        records = records[: args.dry_run_limit]

    overall_file_rows = []
    overall_flag_rows = []
    by_language_file_rows = []
    by_language_flag_rows = []
    bucket_rows_model_conf = []
    bucket_rows_logprob_conf = []
    entropy_rows = []
    summary_path = OUTPUT_DIR / "summary.json"
    all_summary = json.loads(summary_path.read_text()) if (not args.dry_run and summary_path.exists()) else {}

    for model_key in model_keys:
        model_dir = MODEL_DIRS[model_key]
        per_file_dir = OUTPUT_DIR / model_key / "per_file"
        t0 = time.time()
        n_scored, n_skipped = 0, 0
        counts_lock = threading.Lock()
        scored_rows: List[dict] = []

        to_score = []
        for record in records:
            cache_path = per_file_dir / f"{record.file_id}.json"
            if not args.force and not args.dry_run and cache_path.exists():
                scored_rows.append(json.loads(cache_path.read_text()))
                n_scored += 1
                continue
            item = build_score_input(record, model_dir)
            if item is None:
                n_skipped += 1
                continue
            to_score.append(item)

        batch_size = max(1, args.batch_size)
        batches = [to_score[i:i + batch_size] for i in range(0, len(to_score), batch_size)]
        logger.info(f"[{model_key}] {len(to_score)} file(s) to score (Gemini-matched) in {len(batches)} batch(es) of up to {batch_size}, {n_scored} already cached")

        def process_batch(b_idx, batch_items):
            nonlocal n_scored
            file_ids = ", ".join(it["file_id"] for it in batch_items)
            logger.info(f"[{model_key} batch {b_idx}/{len(batches)}] {len(batch_items)} file(s): {file_ids}")
            pairs_by_file = match_files_batch(client, batch_items, logger)
            for item in batch_items:
                result = score_file(item, pairs_by_file.get(item["file_id"], []))
                if not args.dry_run:
                    cache_path = per_file_dir / f"{item['file_id']}.json"
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                with counts_lock:
                    scored_rows.append(result)
                    n_scored += 1

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_batch, i, batch) for i, batch in enumerate(batches, 1)]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[{model_key}] batch worker failed: {exc}\n{traceback.format_exc()}")

        logger.info(f"[{model_key}] done in {time.time() - t0:.1f}s — scored={n_scored} skipped(no gt/no result)={n_skipped}")

        if args.dry_run:
            for r in scored_rows[:5]:
                print(json.dumps(r, ensure_ascii=False, indent=2))
            continue

        def language_block(lang_rows: List[dict]) -> dict:
            return {
                "n_files": len(lang_rows),
                "file_level": aggregate_file_level(lang_rows),
                "flag_level": aggregate_flag_level(lang_rows),
                "confidence_buckets": {
                    "model_confidence": confidence_bucket_table(lang_rows, "model_confidence"),
                    "logprob_derived_confidence": confidence_bucket_table(lang_rows, "logprob_derived_confidence"),
                },
                "entropy_by_correctness": entropy_by_correctness(lang_rows),
            }

        overall_file_rows.append({"model": model_key, **aggregate_file_level(scored_rows)})
        overall_flag_rows.append({"model": model_key, **aggregate_flag_level(scored_rows)})

        by_lang = defaultdict(list)
        for r in scored_rows:
            by_lang[r["language"]].append(r)

        model_summary = {"overall": language_block(scored_rows), "by_language": {}}

        for lang, lang_rows in sorted(by_lang.items()):
            by_language_file_rows.append({"model": model_key, "language": lang, **aggregate_file_level(lang_rows)})
            by_language_flag_rows.append({"model": model_key, "language": lang, **aggregate_flag_level(lang_rows)})
            for row in confidence_bucket_table(lang_rows, "model_confidence"):
                bucket_rows_model_conf.append({"model": model_key, "language": lang, **row})
            for row in confidence_bucket_table(lang_rows, "logprob_derived_confidence"):
                bucket_rows_logprob_conf.append({"model": model_key, "language": lang, **row})
            entropy_rows.append({"model": model_key, "language": lang, **entropy_by_correctness(lang_rows)})
            model_summary["by_language"][lang] = language_block(lang_rows)

        for row in confidence_bucket_table(scored_rows, "model_confidence"):
            bucket_rows_model_conf.append({"model": model_key, "language": "ALL", **row})
        for row in confidence_bucket_table(scored_rows, "logprob_derived_confidence"):
            bucket_rows_logprob_conf.append({"model": model_key, "language": "ALL", **row})
        entropy_rows.append({"model": model_key, "language": "ALL", **entropy_by_correctness(scored_rows)})

        all_summary[model_key] = model_summary
        (OUTPUT_DIR / model_key).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / model_key / "summary.json").write_text(json.dumps(model_summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    merge_and_write_csv(OUTPUT_DIR / "confusion_file_level_overall.csv", overall_file_rows,
              ["model", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_flag_level_overall.csv", overall_flag_rows,
              ["model", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_file_level_by_language.csv", by_language_file_rows,
              ["model", "language", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_flag_level_by_language.csv", by_language_flag_rows,
              ["model", "language", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confidence_buckets_model_confidence.csv", bucket_rows_model_conf,
              ["model", "language", "bucket", "n_flags", "tp", "fp", "precision", "mean_entropy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confidence_buckets_logprob_confidence.csv", bucket_rows_logprob_conf,
              ["model", "language", "bucket", "n_flags", "tp", "fp", "precision", "mean_entropy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "entropy_by_correctness.csv", entropy_rows,
              ["model", "language", "mean_entropy_tp_flags", "n_tp_flags_with_entropy", "mean_entropy_fp_flags", "n_fp_flags_with_entropy"], model_keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("FILE-LEVEL CONFUSION MATRIX (overall)")
    print("=" * 80)
    for row in overall_file_rows:
        print(row)
    print(f"\nFull per-model, per-language results (precision/recall/specificity/accuracy,")
    print(f"confidence-bucket precision, entropy) saved to:")
    print(f"  {OUTPUT_DIR}/summary.json  (everything, nested by model -> language)")
    print(f"  {OUTPUT_DIR}/{{model}}/summary.json  (one model at a time)")

    print("\n" + "=" * 80)
    print("FLAG-LEVEL CONFUSION MATRIX (overall)")
    print("=" * 80)
    for row in overall_flag_rows:
        print(row)

    print("\n" + "=" * 80)
    print("MEAN TOKEN ENTROPY: TP FLAGS vs FP FLAGS (overall)")
    print("=" * 80)
    for row in entropy_rows:
        if row.get("language") == "ALL":
            print(f"{row['model']:32} mean_entropy_tp={row['mean_entropy_tp_flags']}  (n={row['n_tp_flags_with_entropy']})   "
                  f"mean_entropy_fp={row['mean_entropy_fp_flags']}  (n={row['n_fp_flags_with_entropy']})")

    print(f"\nAll CSVs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
