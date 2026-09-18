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

Disambiguation: when a file has more than one ground-truth OR model flag
in the SAME category, a single (gt, model) pair can't be assumed to
correspond just because the category matches — an LLM judges which
specific flags refer to the same quote/incident. When there's exactly one
flag on each side for a category, that pairing is assumed by construction
(no LLM call needed).

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


DISAMBIGUATE_INSTRUCTION = """
You are matching two lists of flagged excerpts from the SAME audio call,
both already known to be the SAME violation category. List A is the
ground-truth (trusted) flags; list B is a candidate model's flags. Decide
which items in A and B refer to the SAME underlying quote/incident in the
call (semantically the same moment, even if the English translation
wording differs) — this is NOT about exact string match, only about
whether they describe the same specific thing being said.

Each A item may match AT MOST one B item and vice versa. Not every item
needs a match (a genuine miss or a genuine extra flag is a valid outcome
you should report, not force a bad match).

Return ONE raw, minified JSON object:
{"pairs": [[a_index, b_index], ...]}
Indices are 0-based, referring to the order items are given below.
If there are no matches, return {"pairs": []}. Your entire response must
be only the minified JSON object and nothing else.
""".strip()

DISAMBIGUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
    },
    "required": ["pairs"],
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


def disambiguate_category_group(client, file_id: str, category: str, gt_idxs: List[int], gt_items: List[dict],
                                 model_idxs: List[int], model_items: List[dict], logger: StageLogger) -> List[Tuple[int, int]]:
    """gt_idxs/model_idxs: original flag indices (into the file's full flags list) for this category.
    gt_items/model_items: same length, the corresponding flag dicts. Returns matched (gt_idx, model_idx) pairs."""
    a_lines = "\n".join(f"A[{i}]: {it['translation']}" for i, it in enumerate(gt_items))
    b_lines = "\n".join(f"B[{i}]: {it['translation']}" for i, it in enumerate(model_items))
    prompt_content = f"{DISAMBIGUATE_INSTRUCTION}\n\nCategory: {category}\n\n[LIST A]\n{a_lines}\n\n[LIST B]\n{b_lines}"

    def do_call():
        return gemini_client.generate_text(client, config.GEMINI_MODEL, contents=[prompt_content], response_json_schema=DISAMBIGUATE_SCHEMA)

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"{file_id} category={category}: disambiguation attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
    parsed = gemini_client.parse_json_lenient(raw_text)

    pairs = []
    used_a, used_b = set(), set()
    for a_i, b_i in parsed.get("pairs", []):
        if a_i in used_a or b_i in used_b:
            continue
        if not (0 <= a_i < len(gt_items)) or not (0 <= b_i < len(model_items)):
            continue
        used_a.add(a_i)
        used_b.add(b_i)
        pairs.append((gt_idxs[a_i], model_idxs[b_i]))
    return pairs


def match_flags(client, file_id: str, gt_flags: List[dict], model_flags: List[dict], logger: StageLogger) -> List[Tuple[int, int]]:
    """Returns matched (gt_index, model_index) pairs across the whole file, grouped per category."""
    gt_by_cat = defaultdict(list)
    for i, f in enumerate(gt_flags):
        gt_by_cat[f["category"]].append(i)
    model_by_cat = defaultdict(list)
    for i, f in enumerate(model_flags):
        model_by_cat[f["category"]].append(i)

    all_pairs = []
    for category in set(gt_by_cat) | set(model_by_cat):
        gt_idxs = gt_by_cat.get(category, [])
        model_idxs = model_by_cat.get(category, [])
        if not gt_idxs or not model_idxs:
            continue  # unmatched on one side entirely -> all FN or all FP for this category, no pairs
        if len(gt_idxs) == 1 and len(model_idxs) == 1:
            all_pairs.append((gt_idxs[0], model_idxs[0]))
            continue
        gt_items = [gt_flags[i] for i in gt_idxs]
        model_items = [model_flags[i] for i in model_idxs]
        all_pairs.extend(disambiguate_category_group(client, file_id, category, gt_idxs, gt_items, model_idxs, model_items, logger))
    return all_pairs


def unique_records():
    seen = set()
    for r in dsv2.load_dataset_v2():
        if r.file_id in seen:
            continue
        seen.add(r.file_id)
        yield r


def score_file(client, record: dsv2.FileRecordV2, model_dir: Path, logger: StageLogger) -> Optional[dict]:
    gt_flags = load_ground_truth(record)
    model_flags = load_model_flags(model_dir, record.file_id)
    if gt_flags is None or model_flags is None:
        return None

    matched_pairs = match_flags(client, record.file_id, gt_flags, model_flags, logger)
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
        "file_id": record.file_id,
        "language": record.language,
        "gt_flag_count": len(gt_flags),
        "model_flag_count": len(model_flags),
        "flag_tp": flag_tp,
        "flag_fn": flag_fn,
        "flag_fp": flag_fp,
        "file_bucket": file_bucket,
        "model_flags": flags_detail,
    }


def confusion_metrics(tp: int, fp: int, fn: int, tn: Optional[int] = None) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}
    if tn is not None:
        out["tn"] = tn
        out["specificity"] = tn / (tn + fp) if (tn + fp) else None
        total = tp + fp + fn + tn
        out["accuracy"] = (tp + tn) / total if total else None
    return out


def aggregate_file_level(rows: List[dict]) -> dict:
    tp = sum(1 for r in rows if r["file_bucket"] == "TP")
    fp = sum(1 for r in rows if r["file_bucket"] == "FP")
    fn = sum(1 for r in rows if r["file_bucket"] == "FN")
    tn = sum(1 for r in rows if r["file_bucket"] == "TN")
    return {"n_files": len(rows), **confusion_metrics(tp, fp, fn, tn)}


def aggregate_flag_level(rows: List[dict]) -> dict:
    tp = sum(r["flag_tp"] for r in rows)
    fp = sum(r["flag_fp"] for r in rows)
    fn = sum(r["flag_fn"] for r in rows)
    return confusion_metrics(tp, fp, fn)


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
            "precision": tp / len(in_bin) if in_bin else None,
            "mean_entropy": sum(entropies) / len(entropies) if entropies else None,
        })
    return table


def entropy_by_correctness(rows: List[dict]) -> dict:
    all_flags = [f for r in rows for f in r["model_flags"]]
    matched = [f["entropy_mean"] for f in all_flags if f["matched"] and f.get("entropy_mean") is not None]
    unmatched = [f["entropy_mean"] for f in all_flags if not f["matched"] and f.get("entropy_mean") is not None]
    return {
        "mean_entropy_tp_flags": sum(matched) / len(matched) if matched else None,
        "n_tp_flags_with_entropy": len(matched),
        "mean_entropy_fp_flags": sum(unmatched) / len(unmatched) if unmatched else None,
        "n_fp_flags_with_entropy": len(unmatched),
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None, help="Comma-separated subset of model keys; default = all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true", help="Recompute per-file scoring even if a cached per-file result exists")
    parser.add_argument("--workers", type=int, default=6)
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

    for model_key in model_keys:
        model_dir = MODEL_DIRS[model_key]
        per_file_dir = OUTPUT_DIR / model_key / "per_file"
        t0 = time.time()
        n_scored, n_skipped = 0, 0
        counts_lock = threading.Lock()
        scored_rows: List[dict] = []

        def process_one(record):
            nonlocal n_scored, n_skipped
            cache_path = per_file_dir / f"{record.file_id}.json"
            if not args.force and not args.dry_run and cache_path.exists():
                result = json.loads(cache_path.read_text())
            else:
                result = score_file(client, record, model_dir, logger)
                if result is None:
                    with counts_lock:
                        n_skipped += 1
                    return
                if not args.dry_run:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            with counts_lock:
                scored_rows.append(result)
                n_scored += 1

        logger.info(f"[{model_key}] scoring {len(records)} file(s) against ground truth...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_one, r) for r in records]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[{model_key}] worker failed: {exc}\n{traceback.format_exc()}")

        logger.info(f"[{model_key}] done in {time.time() - t0:.1f}s — scored={n_scored} skipped(no gt/no result)={n_skipped}")

        if args.dry_run:
            for r in scored_rows[:5]:
                print(json.dumps(r, ensure_ascii=False, indent=2))
            continue

        overall_file_rows.append({"model": model_key, **aggregate_file_level(scored_rows)})
        overall_flag_rows.append({"model": model_key, **aggregate_flag_level(scored_rows)})

        by_lang = defaultdict(list)
        for r in scored_rows:
            by_lang[r["language"]].append(r)
        for lang, lang_rows in sorted(by_lang.items()):
            by_language_file_rows.append({"model": model_key, "language": lang, **aggregate_file_level(lang_rows)})
            by_language_flag_rows.append({"model": model_key, "language": lang, **aggregate_flag_level(lang_rows)})

        for row in confidence_bucket_table(scored_rows, "model_confidence"):
            bucket_rows_model_conf.append({"model": model_key, **row})
        for row in confidence_bucket_table(scored_rows, "logprob_derived_confidence"):
            bucket_rows_logprob_conf.append({"model": model_key, **row})

        entropy_rows.append({"model": model_key, **entropy_by_correctness(scored_rows)})

    if args.dry_run:
        return

    write_csv(OUTPUT_DIR / "confusion_file_level_overall.csv", overall_file_rows,
              ["model", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"])
    write_csv(OUTPUT_DIR / "confusion_flag_level_overall.csv", overall_flag_rows,
              ["model", "tp", "fp", "fn", "precision", "recall"])
    write_csv(OUTPUT_DIR / "confusion_file_level_by_language.csv", by_language_file_rows,
              ["model", "language", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"])
    write_csv(OUTPUT_DIR / "confusion_flag_level_by_language.csv", by_language_flag_rows,
              ["model", "language", "tp", "fp", "fn", "precision", "recall"])
    write_csv(OUTPUT_DIR / "confidence_buckets_model_confidence.csv", bucket_rows_model_conf,
              ["model", "bucket", "n_flags", "tp", "fp", "precision", "mean_entropy"])
    write_csv(OUTPUT_DIR / "confidence_buckets_logprob_confidence.csv", bucket_rows_logprob_conf,
              ["model", "bucket", "n_flags", "tp", "fp", "precision", "mean_entropy"])
    write_csv(OUTPUT_DIR / "entropy_by_correctness.csv", entropy_rows,
              ["model", "mean_entropy_tp_flags", "n_tp_flags_with_entropy", "mean_entropy_fp_flags", "n_fp_flags_with_entropy"])

    print("\n" + "=" * 80)
    print("FILE-LEVEL CONFUSION MATRIX (overall)")
    print("=" * 80)
    for row in overall_file_rows:
        print(row)

    print("\n" + "=" * 80)
    print("FLAG-LEVEL CONFUSION MATRIX (overall)")
    print("=" * 80)
    for row in overall_flag_rows:
        print(row)

    print(f"\nAll CSVs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
