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

MATCHING — a deterministic pre-pass plus a single-file Gemini leftover call,
adopted after an isolated-vs-batched audit found an earlier all-Gemini,
5-files-per-call approach disagreed with a single-file rematch on 13% of a
100-file sample (that earlier approach and its output are no longer kept
around):

  1. DETERMINISTIC PRE-PASS (deterministic_prematch, pure Python, zero LLM
     calls, 100% reproducible by construction): a ground-truth flag and a
     model flag are auto-paired if they share the SAME category, their
     timestamps are within +-15s of each other, AND their native-language
     excerpts have high text similarity (difflib ratio). This resolves the
     easy, unambiguous majority of matches without ever asking Gemini, which
     both cuts cost and removes the LLM's main source of variance for them.
  2. Only the LEFTOVER flags neither side of the pre-pass could pair are
     sent to Gemini — ONE FILE PER CALL (batch size 1 by default, temperature
     0, model config.GEMINI_MATCH_MODEL — a stronger tier than the
     ground-truth generator, not the same size/generation), for whichever
     files still have genuine ambiguity on BOTH sides after the pre-pass.
     Gemini decides which LEFTOVER GT item and LEFTOVER MODEL item refer to
     the same incident (same category AND same semantic content, judged by
     translation) — this is where real judgment calls remain, and only
     there. If --match-runs > 1, each such file's leftover set is matched
     that many times independently and resolved by majority vote per pair,
     with the per-run agreement recorded as that file's match_reliability
     (1.0 for files with no leftovers, or for --match-runs 1).
  3. Python only turns the pre-pass + Gemini pairs into TP/FP/FN counts — no
     arithmetic judgment calls are made by Gemini, only the matching
     decision, and only on the flags the deterministic pass couldn't
     resolve on its own.

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
import difflib
import json
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
import dataset_v2 as dsv2
import gemini_client
from pipeline_logging import StageLogger

# --- Deterministic pre-pass thresholds (see module docstring, MATCHING) ---
PREMATCH_MAX_TS_DIFF_SEC = 15.0
PREMATCH_MIN_TEXT_SIMILARITY = 0.5

def _build_model_dirs(gemma_results_dir: Path, gemini_results_dir: Path) -> Dict[str, Path]:
    return {
        "e2b_thinking": gemma_results_dir / "e2b_thinking",
        "e2b_nothinking": gemma_results_dir / "e2b_nothinking",
        "e4b_thinking": gemma_results_dir / "e4b_thinking",
        "e4b_nothinking": gemma_results_dir / "e4b_nothinking",
        "12b_nothinking": gemma_results_dir / "12b_nothinking",
        "e2b_conformer5s_text_nothinking": gemma_results_dir / "e2b_conformer5s_text_nothinking",
        "e4b_conformer5s_text_nothinking": gemma_results_dir / "e4b_conformer5s_text_nothinking",
        "gemini-3.5-flash-lite": gemini_results_dir / "gemini-3.5-flash-lite",
    }


# DATASET_DIR / MODEL_DIRS / OUTPUT_DIR are mutable module globals, not
# frozen constants: configure_dataset_root() below repoints them at an
# alternate dataset root's OWN result directories (see config.dataset_paths)
# when --dataset-root is passed. Every function in this module (and
# classify_fps.py, which imports this module and reads these same names)
# looks these up at CALL time via attribute access, so calling
# configure_dataset_root() once at the top of main() — before anything else
# runs — is sufficient; no other function needs to change.
DATASET_DIR: Path = config.DOSTT_DIR
MODEL_DIRS: Dict[str, Path] = _build_model_dirs(config.PROJECT_ROOT / "gemma_results", config.PROJECT_ROOT / "gemini_results")
OUTPUT_DIR: Path = config.PROJECT_ROOT / "analysis_results"


def configure_dataset_root(dataset_root: Optional[str] = None):
    """Repoints DATASET_DIR/MODEL_DIRS/OUTPUT_DIR at dataset_root's own
    result directories. Call once, first thing, in any script's main() that
    accepts --dataset-root — including classify_fps.py, via
    analyze_results.configure_dataset_root(args.dataset_root), so both
    scripts resolve to the same dev/prod directories for the same value."""
    global DATASET_DIR, MODEL_DIRS, OUTPUT_DIR
    paths = config.dataset_paths(dataset_root)
    DATASET_DIR = paths["dataset_dir"]
    MODEL_DIRS = _build_model_dirs(paths["gemma_results_dir"], paths["gemini_results_dir"])
    OUTPUT_DIR = paths["analysis_results_dir"]

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
        {
            "category": f["ground_truth_category"],
            "translation": f["ground_truth_translation"],
            "timestamp": f.get("ground_truth_timestamp", ""),
            "excerpt": f.get("ground_truth_excerpt", ""),
            "confidence": f.get("ground_truth_confidence"),
        }
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
            "timestamp": f.get("model_timestamp", ""),
            "excerpt": f.get("model_excerpt", ""),
            "model_confidence": f.get("model_confidence"),
            "logprob_derived_confidence": f.get("logprob_derived_confidence"),
            "entropy_mean": entropy.get("mean"),
        })
    return out


def _parse_ts_sec(ts: str) -> Optional[float]:
    try:
        parts = [float(p) for p in ts.strip().split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return None
    return None


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def deterministic_prematch(gt_flags: List[dict], model_flags: List[dict]) -> List[Tuple[int, int]]:
    """Zero-LLM, fully reproducible pairing: same category, timestamps within
    PREMATCH_MAX_TS_DIFF_SEC, and native-excerpt text similarity above
    PREMATCH_MIN_TEXT_SIMILARITY. Greedy, highest-similarity-first assignment
    so each flag is used at most once. Deliberately conservative (misses are
    fine — they just fall through to the LLM leftover pass) since a WRONG
    deterministic pair here would silently corrupt the baseline with no
    judgment call to catch it."""
    candidates = []
    for gi, g in enumerate(gt_flags):
        g_ts = _parse_ts_sec(g.get("timestamp", ""))
        if g_ts is None:
            continue
        for mi, m in enumerate(model_flags):
            if g.get("category") != m.get("category"):
                continue
            m_ts = _parse_ts_sec(m.get("timestamp", ""))
            if m_ts is None or abs(g_ts - m_ts) > PREMATCH_MAX_TS_DIFF_SEC:
                continue
            sim = text_similarity(g.get("excerpt", ""), m.get("excerpt", ""))
            if sim < PREMATCH_MIN_TEXT_SIMILARITY:
                continue
            candidates.append((sim, gi, mi))

    candidates.sort(key=lambda c: -c[0])
    used_gt, used_model, pairs = set(), set(), []
    for sim, gi, mi in candidates:
        if gi in used_gt or mi in used_model:
            continue
        used_gt.add(gi)
        used_model.add(mi)
        pairs.append((gi, mi))
    return pairs


def match_files_batch(client, batch_items: List[dict], logger: StageLogger,
                       model: str = config.GEMINI_MATCH_MODEL) -> Dict[str, List[Tuple[int, int]]]:
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
        return gemini_client.generate_text(client, model, contents=[prompt_content], response_json_schema=BATCH_MATCH_SCHEMA)

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
    for r in dsv2.load_dataset_v2(DATASET_DIR):
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


def resolve_matches(client, batch_items: List[dict], match_model: str, match_runs: int,
                     logger: StageLogger) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, float]]:
    """For each item in batch_items ({file_id, gt_flags, model_flags}): run the
    deterministic pre-pass first, then send ONLY the leftover flags neither
    side of the pre-pass paired to Gemini — one call for the whole
    batch_items list (batch_items is normally length 1, since analyze_results
    defaults --batch-size to 1; kept generic since match_files_batch already
    supports N files per call).

    If match_runs > 1, the leftover set for each file is matched that many
    times independently and pairs are kept by majority vote (accepted if
    proposed in > half the runs); match_reliability records, per file, the
    fraction of the match_runs whose raw pair-SET exactly equalled the
    majority result (1.0 for files with no leftovers, or when match_runs==1
    — nothing to compare against without a repeat).

    Returns (pairs_by_file, reliability_by_file), both keyed by file_id, with
    every batch_items file_id present in both (reliability defaults to 1.0)."""
    pairs_by_file: Dict[str, List[Tuple[int, int]]] = {}
    reliability_by_file: Dict[str, float] = {}
    leftover_items = []
    leftover_maps: Dict[str, dict] = {}

    for item in batch_items:
        pre_pairs = deterministic_prematch(item["gt_flags"], item["model_flags"])
        pairs_by_file[item["file_id"]] = list(pre_pairs)
        reliability_by_file[item["file_id"]] = 1.0

        used_gt = {p[0] for p in pre_pairs}
        used_model = {p[1] for p in pre_pairs}
        gt_map = [i for i in range(len(item["gt_flags"])) if i not in used_gt]
        model_map = [i for i in range(len(item["model_flags"])) if i not in used_model]
        if gt_map and model_map:  # genuine ambiguity remains on BOTH sides — needs the LLM
            leftover_items.append({
                "file_id": item["file_id"],
                "gt_flags": [item["gt_flags"][i] for i in gt_map],
                "model_flags": [item["model_flags"][i] for i in model_map],
            })
            leftover_maps[item["file_id"]] = {"gt": gt_map, "model": model_map}

    if not leftover_items:
        return pairs_by_file, reliability_by_file

    def safe_match_call():
        try:
            return match_files_batch(client, leftover_items, logger, model=match_model), True
        except Exception as exc:  # noqa: BLE001 — a blocked/failed leftover-match call (e.g.
            # PROHIBITED_CONTENT, or any other API failure) must never crash the whole run;
            # falling back to prepass-only pairs for these files (reliability 0.0, logged
            # loudly) is far safer than an unhandled crash mid-batch.
            file_ids = ", ".join(it["file_id"] for it in leftover_items)
            logger.error(f"leftover match call failed for [{file_ids}]: {exc} — "
                         f"falling back to prepass-only pairs for these file(s), match_reliability=0.0")
            return {}, False

    n_runs = max(1, match_runs)
    if n_runs == 1:
        run_results = [safe_match_call()]
    else:
        with ThreadPoolExecutor(max_workers=n_runs) as run_pool:
            run_results = list(run_pool.map(lambda _: safe_match_call(), range(n_runs)))
    runs = [r for r, _ in run_results]
    any_call_failed = any(not ok for _, ok in run_results)

    for file_id, maps in leftover_maps.items():
        gt_map, model_map = maps["gt"], maps["model"]
        run_pair_sets = [frozenset(runs[r].get(file_id, [])) for r in range(len(runs))]

        if any_call_failed:
            majority_pairs = frozenset()
            for pair_set, (_, ok) in zip(run_pair_sets, run_results):
                if ok:  # keep pairs from whichever runs DID succeed, rather than discarding them
                    majority_pairs = pair_set
                    break
            reliability_by_file[file_id] = 0.0
        elif len(runs) == 1:
            majority_pairs = run_pair_sets[0]
            reliability_by_file[file_id] = 1.0
        else:
            vote_counts = Counter()
            for pair_set in run_pair_sets:
                for pair in pair_set:
                    vote_counts[pair] += 1
            threshold = len(runs) / 2.0
            # Greedy, highest-vote-first so majority-approved pairs still can't double-use a flag.
            used_gt_local, used_model_local, majority_list = set(), set(), []
            for (a_i, b_i), votes in sorted(vote_counts.items(), key=lambda kv: -kv[1]):
                if votes <= threshold or a_i in used_gt_local or b_i in used_model_local:
                    continue
                used_gt_local.add(a_i)
                used_model_local.add(b_i)
                majority_list.append((a_i, b_i))
            majority_pairs = frozenset(majority_list)
            reliability_by_file[file_id] = sum(1 for ps in run_pair_sets if ps == majority_pairs) / len(runs)

        remapped = [(gt_map[a_i], model_map[b_i]) for a_i, b_i in majority_pairs]
        pairs_by_file[file_id].extend(remapped)

    return pairs_by_file, reliability_by_file


# Additive file-level bucket (does NOT replace file_bucket — explicit user
# decision): TP if EVERY ground-truth flag in these two categories, at
# confidence >= QUALIFYING_MIN_CONFIDENCE, has a matched model flag;
# SuspiciousActivity is ignored entirely for this rule (also explicit). Every
# other case (no qualifying GT flags at all, e.g. SA-only or low-confidence
# files; GT empty; a qualifying set that's only PARTIALLY caught) falls back
# to the exact same rule file_bucket already uses — confirmed by explicit
# user answer that a partial catch of qualifying flags should still be TP
# via the old any-match rule, not downgraded to FN. Net effect (verified
# empirically, not just by inspection): identical to file_bucket in every
# case, because "all qualifying flags matched" always implies flag_tp>0
# (already old-rule TP), and the fallback for every other case IS the old
# rule. Kept as its own field/CSV anyway, per explicit request to compute
# and save it, and as a place to plug in a genuinely different rule later.
QUALIFYING_CATEGORIES = {"PlatformMove", "Explicit-Flirting"}
QUALIFYING_MIN_CONFIDENCE = 0.8


def score_file(item: dict, matched_pairs: List[Tuple[int, int]], match_reliability: float = 1.0) -> dict:
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

    qualifying_idx = [
        i for i, f in enumerate(gt_flags)
        if f.get("category") in QUALIFYING_CATEGORIES and (f.get("confidence") or 0) >= QUALIFYING_MIN_CONFIDENCE
    ]
    if qualifying_idx and all(i in matched_gt for i in qualifying_idx):
        qualifying_pm_ef_bucket = "TP"
    else:
        qualifying_pm_ef_bucket = file_bucket  # fallback = the exact same rule as file_bucket

    gt_flags_detail = []
    for i, f in enumerate(gt_flags):
        gt_flags_detail.append({**f, "matched": i in matched_gt})

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
        "qualifying_pm_ef_bucket": qualifying_pm_ef_bucket,
        "match_reliability": r2(match_reliability),
        "gt_flags": gt_flags_detail,
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


def aggregate_file_level(rows: List[dict], bucket_key: str = "file_bucket") -> dict:
    tp = sum(1 for r in rows if r[bucket_key] == "TP")
    fp = sum(1 for r in rows if r[bucket_key] == "FP")
    fn = sum(1 for r in rows if r[bucket_key] == "FN")
    tn = sum(1 for r in rows if r[bucket_key] == "TN")
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


def _mean_entropy(flags: List[dict]) -> Optional[float]:
    vals = [f["entropy_mean"] for f in flags if f.get("entropy_mean") is not None]
    return r2(sum(vals) / len(vals) if vals else None)


def confidence_bucket_table(rows: List[dict], confidence_key: str) -> List[dict]:
    all_flags = [f for r in rows for f in r["model_flags"] if f.get(confidence_key) is not None]
    table = []
    for lo, hi in CONFIDENCE_BINS:
        in_bin = [f for f in all_flags if lo <= f[confidence_key] < hi]
        tp_flags = [f for f in in_bin if f["matched"]]
        fp_flags = [f for f in in_bin if not f["matched"]]
        table.append({
            "bucket": bucket_label(lo, hi),
            "n_flags": len(in_bin),
            "tp": len(tp_flags),
            "fp": len(fp_flags),
            "precision": r2(len(tp_flags) / len(in_bin) if in_bin else None),
            "mean_entropy_overall": _mean_entropy(in_bin),
            "mean_entropy_tp": _mean_entropy(tp_flags),
            "mean_entropy_fp": _mean_entropy(fp_flags),
        })
    return table


def entropy_by_correctness(rows: List[dict]) -> dict:
    """Mean excerpt-token entropy split by whether the flag was matched
    (TP) or not (FP) — higher entropy on FP flags would mean the model was
    quoting text it wasn't actually sure about. Also reports the overall
    mean across every flag regardless of correctness."""
    all_flags = [f for r in rows for f in r["model_flags"]]
    matched = [f for f in all_flags if f["matched"]]
    unmatched = [f for f in all_flags if not f["matched"]]
    return {
        "mean_entropy_overall": _mean_entropy(all_flags),
        "n_flags_with_entropy": sum(1 for f in all_flags if f.get("entropy_mean") is not None),
        "mean_entropy_tp_flags": _mean_entropy(matched),
        "n_tp_flags_with_entropy": sum(1 for f in matched if f.get("entropy_mean") is not None),
        "mean_entropy_fp_flags": _mean_entropy(unmatched),
        "n_fp_flags_with_entropy": sum(1 for f in unmatched if f.get("entropy_mean") is not None),
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


def run_stability_check(client, records, model_key: str, match_model: str, sample_size: int, logger: StageLogger):
    """Samples files that have >=1 leftover flag after the deterministic
    pre-pass (files fully resolved by the pre-pass are 100% reproducible by
    construction and would trivially inflate the agreement rate), runs the
    leftover LLM matching TWICE independently, and reports how often the two
    runs produce the EXACT same pair-set per file. Decides whether a
    production run should use --match-runs 1 (>=98% agreement) or 3
    (majority vote) — writes nothing to analysis_results/."""
    import random

    model_dir = MODEL_DIRS[model_key]
    candidates = []
    for record in records:
        item = build_score_input(record, model_dir)
        if item is None:
            continue
        pre_pairs = deterministic_prematch(item["gt_flags"], item["model_flags"])
        used_gt = {p[0] for p in pre_pairs}
        used_model = {p[1] for p in pre_pairs}
        has_leftover = any(i not in used_gt for i in range(len(item["gt_flags"]))) and \
                        any(i not in used_model for i in range(len(item["model_flags"])))
        if has_leftover:
            candidates.append(item)

    logger.info(f"[stability-check] {len(candidates)} file(s) have a genuine post-prepass leftover "
                f"(out of {len(records)} total) — sampling up to {sample_size}")
    random.seed(42)
    sample = random.sample(candidates, min(sample_size, len(candidates)))

    # One file per call, matching real production behavior (--batch-size 1
    # default) — NOT one resolve_matches(sample) call, which would silently
    # re-bundle all sampled files into a single multi-file match call and
    # reintroduce exactly the batching dilution this fix is meant to remove.
    def run_once(item):
        pairs, _ = resolve_matches(client, [item], match_model, 1, logger)
        return frozenset(pairs.get(item["file_id"], []))

    with ThreadPoolExecutor(max_workers=6) as pool:
        run1 = dict(zip((it["file_id"] for it in sample), pool.map(run_once, sample)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        run2 = dict(zip((it["file_id"] for it in sample), pool.map(run_once, sample)))

    agree = 0
    disagreements = []
    for item in sample:
        fid = item["file_id"]
        s1, s2 = run1.get(fid, frozenset()), run2.get(fid, frozenset())
        if s1 == s2:
            agree += 1
        else:
            disagreements.append((fid, sorted(s1), sorted(s2)))

    rate = 100 * agree / len(sample) if sample else 100.0
    print(f"\nSTABILITY CHECK: {len(sample)} file(s) with genuine leftover ambiguity, model={match_model}")
    print(f"Run 1 vs run 2 exact-pairset agreement: {agree}/{len(sample)} ({rate:.1f}%)")
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):")
        for fid, s1, s2 in disagreements[:20]:
            print(f"  {fid}: run1={s1} run2={s2}")
    if rate >= 98.0:
        print("\n>=98% agreement -> --match-runs 1 (single run) is fine for the real re-score.")
    else:
        print(f"\n<98% agreement -> use --match-runs 3 (majority vote) for the real re-score.")
    logger.info(f"[stability-check] agreement={rate:.1f}% over {len(sample)} sampled file(s), {len(disagreements)} disagreement(s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None, help="Comma-separated subset of model keys; default = all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true", help="Recompute per-file scoring even if a cached per-file result exists")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1, help="Files' worth of LEFTOVER (post-deterministic-prepass) GT+model flags sent to Gemini per matching call (default: one file per call)")
    parser.add_argument("--match-model", type=str, default=config.GEMINI_MATCH_MODEL, help="Gemini model for the leftover matching call")
    parser.add_argument("--match-runs", type=int, default=1, help="Independent leftover-matching runs per file, majority-voted if >1 (see --stability-check to decide this)")
    parser.add_argument("--stability-check", action="store_true", help="Sample --stability-sample files, run leftover matching twice, report per-file agreement rate; writes nothing to analysis_results/")
    parser.add_argument("--stability-sample", type=int, default=100)
    parser.add_argument("--dataset-root", type=str, default=None,
                         help="Alternate dataset root, e.g. Dostt_dev — routes results to "
                              "analysis_results_<suffix>/ (and reads gemma/gemini_results_<suffix>/) "
                              "automatically; default (unset) uses the full Dostt/ dataset")
    args = parser.parse_args()
    configure_dataset_root(args.dataset_root)  # first thing — everything below reads MODEL_DIRS/OUTPUT_DIR/DATASET_DIR

    model_keys = args.models.split(",") if args.models else list(MODEL_DIRS)
    logger = StageLogger("analyze_results")
    client = gemini_client.get_client()

    records = list(unique_records())
    if args.dry_run:
        records = records[: args.dry_run_limit]

    if args.stability_check:
        run_stability_check(client, records, model_keys[0], args.match_model, args.stability_sample, logger)
        return

    overall_file_rows = []
    overall_flag_rows = []
    by_language_file_rows = []
    by_language_flag_rows = []
    qualifying_overall_rows = []
    qualifying_by_language_rows = []
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
            pairs_by_file, reliability_by_file = resolve_matches(client, batch_items, args.match_model, args.match_runs, logger)
            for item in batch_items:
                result = score_file(item, pairs_by_file.get(item["file_id"], []), reliability_by_file.get(item["file_id"], 1.0))
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

        overall_entropy = entropy_by_correctness(scored_rows)
        overall_file_rows.append({"model": model_key, **aggregate_file_level(scored_rows), **overall_entropy})
        overall_flag_rows.append({"model": model_key, **aggregate_flag_level(scored_rows), **overall_entropy})
        qualifying_overall_rows.append({"model": model_key, **aggregate_file_level(scored_rows, "qualifying_pm_ef_bucket")})

        by_lang = defaultdict(list)
        for r in scored_rows:
            by_lang[r["language"]].append(r)

        model_summary = {"overall": language_block(scored_rows), "by_language": {}}

        for lang, lang_rows in sorted(by_lang.items()):
            lang_entropy = entropy_by_correctness(lang_rows)
            by_language_file_rows.append({"model": model_key, "language": lang, **aggregate_file_level(lang_rows), **lang_entropy})
            by_language_flag_rows.append({"model": model_key, "language": lang, **aggregate_flag_level(lang_rows), **lang_entropy})
            qualifying_by_language_rows.append({"model": model_key, "language": lang, **aggregate_file_level(lang_rows, "qualifying_pm_ef_bucket")})
            for row in confidence_bucket_table(lang_rows, "model_confidence"):
                bucket_rows_model_conf.append({"model": model_key, "language": lang, **row})
            for row in confidence_bucket_table(lang_rows, "logprob_derived_confidence"):
                bucket_rows_logprob_conf.append({"model": model_key, "language": lang, **row})
            entropy_rows.append({"model": model_key, "language": lang, **lang_entropy})
            model_summary["by_language"][lang] = language_block(lang_rows)

        for row in confidence_bucket_table(scored_rows, "model_confidence"):
            bucket_rows_model_conf.append({"model": model_key, "language": "ALL", **row})
        for row in confidence_bucket_table(scored_rows, "logprob_derived_confidence"):
            bucket_rows_logprob_conf.append({"model": model_key, "language": "ALL", **row})
        entropy_rows.append({"model": model_key, "language": "ALL", **overall_entropy})

        all_summary[model_key] = model_summary
        (OUTPUT_DIR / model_key).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / model_key / "summary.json").write_text(json.dumps(model_summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    FILE_LEVEL_FIELDS_EXTRA = ["mean_entropy_overall", "n_flags_with_entropy",
                               "mean_entropy_tp_flags", "n_tp_flags_with_entropy",
                               "mean_entropy_fp_flags", "n_fp_flags_with_entropy"]

    merge_and_write_csv(OUTPUT_DIR / "confusion_file_level_overall.csv", overall_file_rows,
              ["model", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"] + FILE_LEVEL_FIELDS_EXTRA, model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_flag_level_overall.csv", overall_flag_rows,
              ["model", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"] + FILE_LEVEL_FIELDS_EXTRA, model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_file_level_by_language.csv", by_language_file_rows,
              ["model", "language", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"] + FILE_LEVEL_FIELDS_EXTRA, model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_file_level_qualifying_pm_ef_overall.csv", qualifying_overall_rows,
              ["model", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_file_level_qualifying_pm_ef_by_language.csv", qualifying_by_language_rows,
              ["model", "language", "n_files", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confusion_flag_level_by_language.csv", by_language_flag_rows,
              ["model", "language", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "accuracy"] + FILE_LEVEL_FIELDS_EXTRA, model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confidence_buckets_model_confidence.csv", bucket_rows_model_conf,
              ["model", "language", "bucket", "n_flags", "tp", "fp", "precision", "mean_entropy_overall", "mean_entropy_tp", "mean_entropy_fp"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "confidence_buckets_logprob_confidence.csv", bucket_rows_logprob_conf,
              ["model", "language", "bucket", "n_flags", "tp", "fp", "precision", "mean_entropy_overall", "mean_entropy_tp", "mean_entropy_fp"], model_keys)
    merge_and_write_csv(OUTPUT_DIR / "entropy_by_correctness.csv", entropy_rows,
              ["model", "language", "mean_entropy_overall", "n_flags_with_entropy",
               "mean_entropy_tp_flags", "n_tp_flags_with_entropy",
               "mean_entropy_fp_flags", "n_fp_flags_with_entropy"], model_keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2))

    print(f"Scored {len(model_keys)} model(s). Results saved to {OUTPUT_DIR}/ "
          f"(summary.json, {{model}}/summary.json, and per-metric CSVs) — no results printed to terminal.")

    print(f"\nAll CSVs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
