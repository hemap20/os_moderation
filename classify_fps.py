"""Sub-classify every unmatched ("FP") model flag by WHY it's wrong.

analyze_results.py already tells you a model flag didn't match any
ground-truth flag ("matched": false in analysis_results/<model>/per_file/
<file_id>.json). This script adds a second, independent layer on top: for
every such flag it asks a Gemini model to walk a fixed 1->9 decision order
(see FP_TYPES below) and assign exactly one fp_type — is the model actually
right and ground truth missed it (GT_MISS), did it invent the quote
(FABRICATION), mishear something real (MISHEARING), flag something the
prompt explicitly excuses (EXCEPTION_IGNORED), etc. This does NOT change any
TP/FP/FN counts or existing CSV columns from analyze_results.py — it is a
purely additive annotation layer.

Every model flag with matched == false is classified, including extra
unmatched flags inside files whose file_bucket is TP (duplicate/extra
instances of an already-matched violation), not just flags inside FP-bucket
files. Each classified flag records file_bucket alongside it so the two
cases can be separated downstream.

The Dostt/<lang>_<cat>/{TP,FP,FN} folder name (record.original_bucket) is
the dataset's ORIGINAL pre-sort label, not the model's outcome — it is
passed to the classifier only as extra context, named dataset_origin_label,
and is never used as (or confused with) the fp_type.

You run this yourself, interactively:

    python3 classify_fps.py --dry-run                              # 3 flags, prints only
    python3 classify_fps.py --models e2b_thinking --languages hindi  # one model, one language
    python3 classify_fps.py                                          # everything
    python3 classify_fps.py --force                                  # ignore cached fp_classification
    python3 classify_fps.py --score-review analysis_results/fp_classification/review_sample.csv
"""
import argparse
import csv
import datetime
import hashlib
import json
import random
import re
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import analyze_results as ar
import config
import dataset_v2 as dsv2
import gemini_client
import prompt_loader
from pipeline_logging import StageLogger

# Mutable module global (not a frozen constant) — reassigned in main() right
# after ar.configure_dataset_root(args.dataset_root), so a --dataset-root
# run's fp_classification/ outputs land under that root's OWN
# analysis_results_<suffix>/ directory, never mixed with full-dataset runs.
OUTPUT_DIR = config.PROJECT_ROOT / "analysis_results" / "fp_classification"
TAXONOMY_VERSION = "fp_v1"

FP_TYPES = [
    "GT_MISS",
    "WRONG_CATEGORY",
    "MATCHER_MISS",
    "DUPLICATE_OR_EXTRA_INSTANCE",
    "FABRICATION",
    "MISHEARING",
    "EXCEPTION_IGNORED",
    "CONTEXT_CONFUSION",
    "OVER_SENSITIVITY",
    "UNCLEAR",
]

RELIABILITY_VALUES = ["ok", "suspect"]
CONFIDENCE_VALUES = ["low", "medium", "high"]

WINDOW_SEC = 90.0
FULL_TRANSCRIPT_CHAR_LIMIT = 6000  # below this, just send the whole transcript instead of windowing

FP_CLASSIFY_INSTRUCTION = """
You are a strict error-type classifier for a content-moderation eval. You
will be given, for SEVERAL flags at once (possibly from different files),
one MODEL FLAG that a candidate model raised which did NOT match any
ground-truth flag (a "false positive" candidate) — plus the full
ground-truth transcript context, all ground-truth flags for that file, the
model's OTHER flags in that file, and the policy definitions/exceptions
below. Your job is to say WHY the model flag is wrong (or, in the
GT_MISS case, why it might actually be right and ground truth is what's
wrong).

Ground truth was itself produced by a Gemini model reading the same
transcript — it can be incomplete or mistaken. You are a stronger check on
it, not an echo of it.

For EACH flag, walk this decision order top to bottom and STOP at the FIRST
type that fits (never assign more than one; earlier checks rule out reasons
that would make later ones meaningless):

1. GT_MISS — the model is actually right and ground truth missed a real
   violation ENTIRELY (no ground-truth flag anywhere in this file covers it):
   the quote is present in the transcript, it clearly meets the category's
   policy definition, and no exception applies. Use this conservatively —
   every GT_MISS you assign gets human-reviewed.
2. WRONG_CATEGORY — the same (or essentially the same) utterance IS flagged
   in ground truth, just under a different category. Content was real and
   violating; only the category is wrong.
3. MATCHER_MISS — the SAME SPECIFIC utterance you're looking at (the same
   words/moment, close in timestamp) IS what a ground-truth flag in this
   file is ALSO describing, but that ground-truth flag is marked
   matched=false below — i.e. the automated scorer failed to pair them, so
   this flag is being counted as a false positive AND that ground-truth
   flag is being counted as a false negative, when really the model caught
   it correctly. This is a scoring/matching artifact, not a model error.
   related_gt_flag_index MUST point at that unmatched (matched=false)
   ground-truth flag.
   STRICT TEST — do not use this type on "same general topic" or "same
   ongoing exchange" grounds. Two different sentences about WhatsApp, or two
   different moments both about moving to Instagram, are DIFFERENT
   instances even if they're part of the same back-and-forth — that is
   DUPLICATE_OR_EXTRA_INSTANCE territory (type 4) if the topic is already
   matched=true elsewhere, or a genuine miss on THIS specific flag if
   nothing covers it at all (types 1/9). Only use MATCHER_MISS when you
   would say "this is the same quote as GT[i], just at a slightly different
   timestamp or translated slightly differently" — not "this is part of the
   same conversation as GT[i]."
   MANDATORY COMPARISON PROCEDURE (do this before choosing between types 3
   and 4 whenever there is more than one GT flag): compare the model flag's
   excerpt/translation against EVERY ground-truth flag's OWN excerpt/
   translation text in this file — not the surrounding transcript lines, not
   "what this part of the conversation is about" — and identify whichever
   ONE ground-truth flag is the closest paraphrase of THIS SPECIFIC model
   flag. Only then look at that one flag's matched status: matched=true ->
   type 4 (DUPLICATE_OR_EXTRA_INSTANCE), matched=false -> type 3
   (MATCHER_MISS). Never pick a ground-truth flag as the "related" one
   because it is temporally closer, or because it's the only unmatched one
   in the file — closeness in CONTENT to that flag's own translation is the
   only criterion. If no ground-truth flag is a reasonable paraphrase of
   this model flag at all, it's neither type 3 nor 4 — move on to type 5+.
4. DUPLICATE_OR_EXTRA_INSTANCE — this flag is ANOTHER, distinct instance of
   a violation that's already matched=true to some OTHER ground-truth flag
   in this file (the same utterance repeated, or another occurrence of what
   ground truth already represented with one flag, e.g. a phone number
   spoken twice). related_gt_flag_index MUST point at that matched=true
   ground-truth flag. Do NOT use this type if the ground-truth flag you're
   pointing at is matched=false — that's type 3 (MATCHER_MISS) instead. This
   is a scoring artifact, not a real model error.
5. FABRICATION — neither the quoted content nor anything resembling it
   appears anywhere in the ground-truth transcript, including near the
   timestamp. The model invented it. evidence_transcript_quote MUST be "" in
   this case.
6. MISHEARING — something similar-sounding or partially overlapping IS in
   the transcript near the timestamp, but the model's version changes the
   meaning in a way that creates the violation (homophone, misheard
   code-mixed word, misheard number/app name). The difference from
   FABRICATION is whether a real source utterance was distorted.
7. EXCEPTION_IGNORED — the quote is real and accurately understood, but it
   matches an explicit "Do NOT flag / NEVER FLAG / Override" item in the
   policy below (e.g. in-platform video-call request, coins/gifts, a denial
   of contact info, "I can't hear you"). Quote which exact exception text in
   fp_exception_matched.
8. CONTEXT_CONFUSION — the quote is real and would be a violation in
   isolation, but context shows it isn't actually happening now
   (hypothetical, reported speech about someone else, a joke, a negation, a
   declined question, an expert warning the user, a description of a
   general practice) — AND is not covered by an explicit exception
   (otherwise it would be type 7).
9. OVER_SENSITIVITY — the quote is real, accurately understood, and in
   context, but doesn't reach the violation threshold (mild romance/
   compliments flagged as Explicit-Flirting, general money talk flagged as
   SuspiciousActivity, sharing a city/name flagged as PlatformMove). No
   explicit exception names it; the category definition just doesn't cover
   it.
10. UNCLEAR — evidence is insufficient to decide: the transcript is missing
    or incomplete around the timestamp, the timestamp is invalid, or the
    quote is too garbled. Always state what's missing in fp_reason.

Remember you are seeing a TRANSCRIPT, not audio. FABRICATION and MISHEARING
are judged against the ground-truth transcript text, which may itself
contain transcription errors — set transcript_reliability to "suspect" (else
"ok") when the transcript itself looks garbled near the timestamp.

IMPORTANT — the matched=true/false shown on each ground-truth flag below is
the AUTHORITATIVE scorer output for THIS file (recomputed fresh, not your
guess). You MUST use it to choose between MATCHER_MISS (3) and
DUPLICATE_OR_EXTRA_INSTANCE (4) — this isn't optional, and a code-level
check will independently re-verify and correct your fp_type if you get types
3/4 backwards, so get it right the first time.

For each flag return an object with these fields, in this order:
- item_id: copy EXACTLY the item_id given for this flag.
- reasoning: string, your step-by-step walk through the decision order
  above, written BEFORE you decide fp_type.
- fp_type: one of GT_MISS | WRONG_CATEGORY | MATCHER_MISS |
  DUPLICATE_OR_EXTRA_INSTANCE | FABRICATION | MISHEARING | EXCEPTION_IGNORED
  | CONTEXT_CONFUSION | OVER_SENSITIVITY | UNCLEAR
- fp_reason: one sentence, English.
- evidence_transcript_quote: the exact transcript text you relied on, or ""
  if nothing was found (required "" for FABRICATION).
- evidence_timestamp: "mm:ss" of that evidence, or "" if none.
- fp_exception_matched: the exact policy exception text you matched (type 7
  only), else "".
- related_gt_flag_index: the 0-based index into that file's GT flag list
  this relates to (types 2/3/4), else null.
- transcript_reliability: "ok" or "suspect".
- classifier_confidence: "low" | "medium" | "high" — your confidence in this
  classification itself.

Return ONE raw, minified JSON object: {"results": [{...}, ...]}. Include
EVERY item_id given, exactly once each. Your entire response must be only
the minified JSON object and nothing else.
""".strip()

FP_CLASSIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "fp_type": {"type": "string", "enum": FP_TYPES},
                    "fp_reason": {"type": "string"},
                    "evidence_transcript_quote": {"type": "string"},
                    "evidence_timestamp": {"type": "string"},
                    "fp_exception_matched": {"type": "string"},
                    "related_gt_flag_index": {"type": ["integer", "null"]},
                    "transcript_reliability": {"type": "string", "enum": RELIABILITY_VALUES},
                    "classifier_confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
                },
                "required": [
                    "item_id", "reasoning", "fp_type", "fp_reason",
                    "evidence_transcript_quote", "evidence_timestamp",
                    "fp_exception_matched", "related_gt_flag_index",
                    "transcript_reliability", "classifier_confidence",
                ],
            },
        },
    },
    "required": ["results"],
}


def r2(x: Optional[float]) -> Optional[float]:
    return round(x, 2) if x is not None else None


def prompt_hash() -> str:
    text = prompt_loader.load_raw_prompt(config.CLASSIFICATION_PROMPT_PATH)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_mmss_to_sec(ts: str) -> Optional[float]:
    try:
        parts = [float(p) for p in ts.strip().split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Loaders — reuse dataset_v2 / analyze_results conventions rather than
# reinventing file layout knowledge.
# ---------------------------------------------------------------------------
def load_full_ground_truth(record: dsv2.FileRecordV2) -> Optional[dict]:
    """Full GT block for a file: transcript segments/full_text/incomplete
    flag, plus every ground_truth_ field per flag (not just category+
    translation, unlike analyze_results.load_ground_truth)."""
    p = record.transcripts_dir / f"{record.file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    transcript = data.get("transcript", {})
    if transcript.get("status") != "success":
        return None
    cls = data.get("classification", {})
    if cls.get("status") != "success":
        return None
    return {
        "segments": transcript.get("segments", []),
        "full_text": transcript.get("full_text", ""),
        "incomplete_transcript": transcript.get("incomplete_transcript", False),
        "gt_flags": cls.get("ground_truth_flags", []),
    }


def load_raw_model_flags(model_dir: Path, file_id: str) -> Optional[List[dict]]:
    """Raw flags list straight from gemma_results/gemini_results, in the same
    order load_model_flags() reads them in — index-aligned with per_file's
    cached model_flags list, since both come from this same "flags" array."""
    p = model_dir / "results" / f"{file_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("status") != "success":
        return None
    return data.get("flags", [])


def transcript_context(gt: dict, timestamp: str, excerpt: str) -> str:
    """Full transcript if short enough, else a +-90s window around the
    timestamp plus any line containing one of the excerpt's key terms."""
    segments = gt["segments"]
    if len(gt["full_text"]) <= FULL_TRANSCRIPT_CHAR_LIMIT or not segments:
        lines = [f'[{s.get("t")}] {s.get("text","")}' for s in segments]
        return "\n".join(lines) if lines else gt["full_text"]

    center = _parse_mmss_to_sec(timestamp)
    key_terms = [w for w in re.split(r"\s+", excerpt) if len(w) >= 3]

    picked = []
    for s in segments:
        t_sec = _parse_mmss_to_sec(s.get("t", ""))
        in_window = center is not None and t_sec is not None and abs(t_sec - center) <= WINDOW_SEC
        matches_term = any(term in s.get("text", "") for term in key_terms)
        if in_window or matches_term:
            picked.append(s)
    if not picked:
        picked = segments
    lines = [f'[{s.get("t")}] {s.get("text","")}' for s in picked]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Work item collection
# ---------------------------------------------------------------------------
class WorkItem:
    __slots__ = (
        "item_id", "model_key", "file_id", "flag_idx", "file_bucket", "language",
        "dataset_origin_label", "category", "timestamp", "excerpt", "translation",
        "justification", "model_confidence", "logprob_derived_confidence",
        "entropy_mean", "gt", "other_model_flags", "matched_gt_indices", "match_verified",
    )

    def __init__(self, **kw):
        self.matched_gt_indices = set()
        self.match_verified = True
        for k, v in kw.items():
            setattr(self, k, v)


def match_key(model_key: str, file_id: str) -> str:
    return f"{model_key}::{file_id}"


def compute_matched_gt_indices(client, match_items: List[dict], batch_size: int,
                                workers: int, logger: StageLogger) -> Dict[str, set]:
    """Reuses analyze_results.match_files_batch (the SAME Gemini-matching
    logic/model analyze_results.py itself scores with) to get the authoritative
    (gt_idx, model_idx) pairs for a set of files, keyed by match_key(model_key,
    file_id). This is what tells DUPLICATE_OR_EXTRA_INSTANCE (points at an
    ALREADY-matched GT flag) apart from MATCHER_MISS (points at a GT flag
    that's real but the scorer failed to pair) — per_file only stores match
    COUNTS (flag_tp/flag_fn), never which GT index matched which model index,
    so this has to be recomputed, not read off disk."""
    if not match_items:
        return {}
    batches = [match_items[i:i + batch_size] for i in range(0, len(match_items), batch_size)]
    result: Dict[str, set] = {}
    lock = threading.Lock()

    def process(b_idx, batch):
        pairs_by_file = ar.match_files_batch(client, batch, logger)
        with lock:
            for key, pairs in pairs_by_file.items():
                result[key] = {p[0] for p in pairs}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, i, b) for i, b in enumerate(batches, 1)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"matched-GT recompute batch failed: {exc}\n{traceback.format_exc()}")
    return result


class EligibilityMismatch(RuntimeError):
    """Raised when the number of flags this script considers eligible for FP
    classification (matched is strictly False) doesn't equal the flag_fp sum
    analyze_results.py already computed over the same file set. A mismatch
    means the two scripts disagree about which flags are unmatched — e.g.
    per_file on disk is stale/from a different run than what's being read
    elsewhere, or a future edit reintroduces the matched-filter bug. Either
    way, classifying is unsafe until the discrepancy is understood, so this
    aborts the whole run rather than silently classifying the wrong flags."""


def collect_work_items(model_keys: List[str], languages: Optional[List[str]],
                        force: bool, cur_prompt_hash: str, classifier_model: str,
                        client, match_batch_size: int, workers: int,
                        logger: StageLogger) -> Dict[str, dict]:
    """Returns (file_state, work_items). file_state maps
    (model_key, file_id) -> {"path": Path, "data": dict} for every file
    touched (even ones with zero pending flags, since excerpt/timestamp
    backfill still dirties them).

    HARD CHECK: eligibility for classification is "matched is False", strictly
    (not "falsy" / missing — a flag with no matched key at all is a data bug,
    not an eligible flag). After each model, the number of flags found
    eligible by that strict rule is asserted equal to the flag_fp analyze_
    results.py already summed into each per_file JSON for the same files;
    a mismatch raises EligibilityMismatch and stops the run immediately,
    because it means this script and analyze_results.py disagree about which
    flags are FPs (stale/mismatched per_file cache, or a re-introduced
    filter bug), and classifying under that disagreement would silently
    tag the wrong flags."""
    records_by_id = {r.file_id: r for r in ar.unique_records()}
    file_state: Dict[tuple, dict] = {}
    work_items: List[WorkItem] = []

    for model_key in model_keys:
        model_dir = ar.MODEL_DIRS[model_key]
        per_file_dir = ar.OUTPUT_DIR / model_key / "per_file"
        if not per_file_dir.is_dir():
            logger.warn(f"[{model_key}] no per_file dir at {per_file_dir} — run analyze_results.py first, skipping")
            continue

        eligible_count = 0
        flag_fp_sum = 0
        n_files_checked = 0
        n_files_skipped = 0

        for cache_path in sorted(per_file_dir.glob("*.json")):
            data = json.loads(cache_path.read_text())
            file_id = data["file_id"]
            record = records_by_id.get(file_id)
            if record is None:
                continue
            if languages and record.language not in languages:
                continue

            raw_flags = load_raw_model_flags(model_dir, file_id)
            gt = load_full_ground_truth(record)
            if raw_flags is None or gt is None:
                n_files_skipped += 1
                continue
            model_flags = data.get("model_flags", [])
            if len(model_flags) != len(raw_flags):
                logger.warn(f"[{model_key}] {file_id}: cached model_flags ({len(model_flags)}) "
                            f"!= raw flags ({len(raw_flags)}); skipping file (rerun analyze_results.py)")
                n_files_skipped += 1
                continue

            n_files_checked += 1
            flag_fp_sum += data.get("flag_fp", 0)

            for i, (mf, rf) in enumerate(zip(model_flags, raw_flags)):
                mf["excerpt"] = rf.get("model_excerpt", "")
                mf["timestamp"] = rf.get("model_timestamp", "")
                mf["dataset_origin_label"] = record.original_bucket

                matched_val = mf.get("matched")
                if matched_val is not False:  # strict: eligible ONLY if explicitly False
                    if matched_val is None:
                        logger.warn(f"[{model_key}] {file_id} flag[{i}]: no 'matched' key at all — "
                                    f"treating as NOT eligible (data bug in per_file cache, not a false positive)")
                    continue  # only unmatched flags get an fp_classification
                eligible_count += 1

                existing = mf.get("fp_classification")
                already_done = (
                    existing
                    and not force
                    and existing.get("taxonomy_version") == TAXONOMY_VERSION
                    and existing.get("prompt_hash") == cur_prompt_hash
                )
                if already_done:
                    continue

                other_flags = [
                    {"category": raw_flags[j].get("model_category", ""),
                     "translation": raw_flags[j].get("model_translation", ""),
                     "matched": model_flags[j].get("matched", False)}
                    for j in range(len(raw_flags)) if j != i
                ]
                work_items.append(WorkItem(
                    item_id=f"{model_key}::{file_id}::{i}",
                    model_key=model_key, file_id=file_id, flag_idx=i,
                    file_bucket=data.get("file_bucket", ""),
                    language=record.language,
                    dataset_origin_label=record.original_bucket,
                    category=rf.get("model_category", ""),
                    timestamp=rf.get("model_timestamp", ""),
                    excerpt=rf.get("model_excerpt", ""),
                    translation=rf.get("model_translation", ""),
                    justification=rf.get("model_justification", ""),
                    model_confidence=rf.get("model_confidence"),
                    logprob_derived_confidence=rf.get("logprob_derived_confidence"),
                    entropy_mean=(rf.get("excerpt_token_entropy") or {}).get("mean"),
                    gt=gt, other_model_flags=other_flags,
                ))
            # excerpt/timestamp/dataset_origin_label backfill always dirties
            # the file even if it has zero flags pending classification. gt/
            # raw_flags are kept too, so the matched-GT recompute below (only
            # for files that actually have pending work) doesn't have to
            # reload/reparse anything.
            file_state[(model_key, file_id)] = {"path": cache_path, "data": data, "gt": gt, "raw_flags": raw_flags}

        if n_files_skipped:
            logger.warn(f"[{model_key}] {n_files_skipped} file(s) skipped (missing raw results/GT, or "
                        f"length mismatch) — excluded from both sides of the eligibility check below")
        logger.info(f"[{model_key}] eligibility check: {eligible_count} flag(s) with matched==False strictly, "
                    f"vs flag_fp sum {flag_fp_sum} over {n_files_checked} checked file(s)")
        if eligible_count != flag_fp_sum:
            raise EligibilityMismatch(
                f"[{model_key}] eligible flag count ({eligible_count}) != flag_fp sum from per_file "
                f"results ({flag_fp_sum}) over {n_files_checked} file(s), language filter={languages}. "
                f"This means the matched==False filter here disagrees with analyze_results.py's own FP "
                f"count — check for a stale/mismatched analysis_results/{model_key}/per_file cache "
                f"(does local analysis_results/ match what's actually pushed/current?), or a bug in the "
                f"eligibility filter. Refusing to classify until this is resolved."
            )

    # Recompute the authoritative matched-GT-index set (via the exact same
    # Gemini matching logic analyze_results.py itself uses) for every file
    # that has >=1 flag pending classification — needed to tell
    # MATCHER_MISS (points at a real but matched=false GT flag) apart from
    # DUPLICATE_OR_EXTRA_INSTANCE (points at a matched=true one). Only done
    # for touched files, not the whole dataset, to keep the extra API cost
    # proportional to the classification job itself.
    touched_keys = sorted({(it.model_key, it.file_id) for it in work_items})
    match_items = []
    for model_key, file_id in touched_keys:
        state = file_state[(model_key, file_id)]
        gt_flags_min = [
            {"category": f.get("ground_truth_category", ""), "translation": f.get("ground_truth_translation", "")}
            for f in state["gt"]["gt_flags"]
        ]
        model_flags_min = [
            {"category": rf.get("model_category", ""), "translation": rf.get("model_translation", "")}
            for rf in state["raw_flags"]
        ]
        match_items.append({"file_id": match_key(model_key, file_id), "gt_flags": gt_flags_min, "model_flags": model_flags_min})

    logger.info(f"recomputing matched-GT indices for {len(match_items)} file(s) with pending work")
    matched_gt_by_file = compute_matched_gt_indices(client, match_items, match_batch_size, workers, logger)

    # HARD CHECK: a multi-file batch call can silently drop or garble one
    # file's result (observed in practice — a file that resolves correctly
    # in an isolated single-file call came back empty inside a 5-file batch).
    # Cross-check each file's recomputed matched-GT count against the
    # trusted flag_tp already on disk from analyze_results.py; on a
    # mismatch, retry that ONE file in isolation (a singleton batch), and if
    # it STILL disagrees, don't trust it — mark it unverified rather than
    # silently feeding a wrong matched-set into the MATCHER_MISS/DUPLICATE
    # code-level rule (which would then confidently enforce the WRONG label).
    unverified_keys = set()
    mismatched_items = []
    for model_key, file_id in touched_keys:
        key = match_key(model_key, file_id)
        expected_tp = file_state[(model_key, file_id)]["data"].get("flag_tp", 0)
        got_tp = len(matched_gt_by_file.get(key, set()))
        if got_tp != expected_tp:
            mismatched_items.append(next(mi for mi in match_items if mi["file_id"] == key))

    if mismatched_items:
        logger.warn(f"{len(mismatched_items)} file(s) disagreed with cached flag_tp after batched "
                    f"matched-GT recompute — retrying each in isolation")
        retried = compute_matched_gt_indices(client, mismatched_items, 1, workers, logger)
        matched_gt_by_file.update(retried)
        for mi in mismatched_items:
            key = mi["file_id"]
            model_key, file_id = key.split("::", 1)
            expected_tp = file_state[(model_key, file_id)]["data"].get("flag_tp", 0)
            got_tp = len(matched_gt_by_file.get(key, set()))
            if got_tp != expected_tp:
                logger.error(f"{key}: matched-GT recompute STILL disagrees with cached flag_tp "
                             f"({got_tp} vs {expected_tp}) after isolated retry — marking UNVERIFIED, "
                             f"MATCHER_MISS/DUPLICATE code-level enforcement disabled for its flags")
                unverified_keys.add(key)

    for it in work_items:
        key = match_key(it.model_key, it.file_id)
        it.matched_gt_indices = matched_gt_by_file.get(key, set())
        it.match_verified = key not in unverified_keys

    return file_state, work_items


def write_backfill(file_state: dict):
    """Persists the excerpt/timestamp/dataset_origin_label fields added to
    every flag (matched or not) during collect_work_items, even for files
    that end up with zero flags pending classification this run."""
    for state in file_state.values():
        state["path"].write_text(json.dumps(state["data"], ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Prompt building / batched classification calls
# ---------------------------------------------------------------------------
def build_item_block(item: WorkItem, policy_text: str) -> str:
    gt_lines = []
    for i, f in enumerate(item.gt["gt_flags"]):
        matched_str = (i in item.matched_gt_indices) if item.match_verified else "UNVERIFIED"
        gt_lines.append(
            f'  GT[{i}] matched={matched_str} '
            f'category={f.get("ground_truth_category")} '
            f'timestamp={f.get("ground_truth_timestamp")} '
            f'excerpt="{f.get("ground_truth_excerpt")}" '
            f'translation="{f.get("ground_truth_translation")}" '
            f'justification="{f.get("ground_truth_justification")}" '
            f'confidence={f.get("ground_truth_confidence")}'
        )
    verification_note = (
        "" if item.match_verified else
        "\nWARNING: matched-status recomputation for this file could not be verified against the "
        "cached scorer output — treat the matched=UNVERIFIED flags above as UNKNOWN. Do NOT use "
        "types 3 (MATCHER_MISS) or 4 (DUPLICATE_OR_EXTRA_INSTANCE) unless you are certain from the "
        "transcript content alone; prefer type 9 (UNCLEAR) if genuinely unsure.\n"
    )
    other_lines = [
        f'  OTHER_MODEL_FLAG[{j}] category={f["category"]} translation="{f["translation"]}" matched={f["matched"]}'
        for j, f in enumerate(item.other_model_flags)
    ]
    transcript = transcript_context(item.gt, item.timestamp, item.excerpt)

    return f"""
[FLAG item_id="{item.item_id}"]
language={item.language}
dataset_origin_label={item.dataset_origin_label}
file_bucket={item.file_bucket}
incomplete_transcript={item.gt["incomplete_transcript"]}{verification_note}

MODEL FLAG (unmatched — classify this one):
  category={item.category}
  timestamp={item.timestamp}
  excerpt="{item.excerpt}"
  translation="{item.translation}"
  justification="{item.justification}"
  model_confidence={item.model_confidence}

ALL GROUND-TRUTH FLAGS FOR THIS FILE:
{chr(10).join(gt_lines) or "  (none)"}

THIS MODEL'S OTHER FLAGS IN THIS FILE:
{chr(10).join(other_lines) or "  (none)"}

GROUND-TRUTH TRANSCRIPT (context):
{transcript or "  (empty)"}
""".strip()


def classify_batch(client, model: str, items: List[WorkItem], policy_text: str, logger: StageLogger) -> Dict[str, dict]:
    blocks = [build_item_block(it, policy_text) for it in items]
    prompt_content = (
        f"{FP_CLASSIFY_INSTRUCTION}\n\n"
        f"[POLICY TEXT — definitions and exceptions to check against]\n{policy_text}\n\n"
        + "\n\n".join(blocks)
    )

    def do_call():
        return gemini_client.generate_text(client, model, contents=[prompt_content], response_json_schema=FP_CLASSIFY_RESPONSE_SCHEMA)

    def on_retry(attempt, max_retries, delay, exc):
        ids = ", ".join(it.item_id for it in items)
        logger.warn(f"batch [{ids}]: classify attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
    parsed = gemini_client.parse_json_lenient(raw_text)

    items_by_id = {it.item_id: it for it in items}
    out = {}
    for entry in parsed.get("results", []):
        item_id = entry.get("item_id")
        if not item_id or not validate_entry(entry):
            continue
        item = items_by_id.get(item_id)
        if item is not None:
            entry = enforce_duplicate_matcher_rule(entry, item, logger)
        out[item_id] = entry
    return out


def enforce_duplicate_matcher_rule(entry: dict, item: WorkItem, logger: Optional[StageLogger] = None) -> dict:
    """Hard, deterministic (non-LLM) safeguard: DUPLICATE_OR_EXTRA_INSTANCE
    is only valid when related_gt_flag_index points at a GT flag the
    authoritative recomputed matcher (item.matched_gt_indices) says IS
    matched=true (matched to some OTHER model flag). If the classifier
    labeled a flag DUPLICATE_OR_EXTRA_INSTANCE but pointed at a GT flag that
    is NOT actually matched — i.e. still a real, uncaught ground-truth
    violation — this overrides the label to MATCHER_MISS instead: the
    content is real and present in ground truth, but analyze_results.py's
    own matcher failed to pair them, which is a scoring artifact, not
    grounds to call this a duplicate. Runs on every DUPLICATE_OR_EXTRA_
    INSTANCE verdict regardless of the classifier's own reasoning. Skipped
    entirely (but flagged low-confidence for human review) when the matched-
    GT recompute for this file couldn't be verified against the cached
    scorer output — enforcing a rule on an unreliable matched-set would be
    worse than not enforcing it at all."""
    if not item.match_verified:
        if entry.get("fp_type") in ("DUPLICATE_OR_EXTRA_INSTANCE", "MATCHER_MISS"):
            entry = dict(entry)
            entry["classifier_confidence"] = "low"
            entry["fp_reason"] = (
                f"{entry.get('fp_reason', '')} [unverified: matched-GT recompute for this file "
                f"disagreed with the cached scorer output and could not be resolved — this "
                f"{entry['fp_type']} call is unconfirmed, needs human review]"
            )
        return entry
    if entry.get("fp_type") != "DUPLICATE_OR_EXTRA_INSTANCE":
        return entry
    rgi = entry.get("related_gt_flag_index")
    if rgi is not None and rgi in item.matched_gt_indices:
        return entry  # correctly points at an already-matched GT flag — stands as-is

    entry = dict(entry)
    original_reason = entry.get("fp_reason", "")
    note = (
        f"related_gt_flag_index={rgi} is not in this file's matched-GT set {sorted(item.matched_gt_indices)} — "
        f"DUPLICATE_OR_EXTRA_INSTANCE requires pointing at an ALREADY-matched GT flag, so this is a "
        f"MATCHER_MISS instead (the content is real ground truth, the scorer just failed to pair it)."
    )
    entry["fp_type"] = "MATCHER_MISS"
    entry["fp_reason"] = f"{original_reason} [code override: {note}]"
    entry["reasoning"] = f"{entry.get('reasoning', '')}\n[CODE OVERRIDE] {note}"
    if logger:
        logger.warn(f"{item.item_id}: classifier said DUPLICATE_OR_EXTRA_INSTANCE pointing at "
                    f"unmatched GT[{rgi}] — overridden to MATCHER_MISS")
    return entry


def validate_entry(entry: dict) -> bool:
    if entry.get("fp_type") not in FP_TYPES:
        return False
    if entry.get("transcript_reliability") not in RELIABILITY_VALUES:
        return False
    if entry.get("classifier_confidence") not in CONFIDENCE_VALUES:
        return False
    if not isinstance(entry.get("evidence_transcript_quote", ""), str):
        return False
    rgi = entry.get("related_gt_flag_index")
    if rgi is not None and not isinstance(rgi, int):
        return False
    return True


def unclear_fallback(reason: str) -> dict:
    return {
        "reasoning": reason,
        "fp_type": "UNCLEAR",
        "fp_reason": reason,
        "evidence_transcript_quote": "",
        "evidence_timestamp": "",
        "fp_exception_matched": "",
        "related_gt_flag_index": None,
        "transcript_reliability": "suspect",
        "classifier_confidence": "low",
    }


def apply_entry(state: dict, it: WorkItem, entry: dict, classifier_model: str, cur_prompt_hash: str, now: str):
    mf = state["data"]["model_flags"][it.flag_idx]
    mf["fp_classification"] = {
        "reasoning": entry["reasoning"],
        "fp_type": entry["fp_type"],
        "fp_reason": entry["fp_reason"],
        "evidence_transcript_quote": entry["evidence_transcript_quote"],
        "evidence_timestamp": entry.get("evidence_timestamp", ""),
        "fp_exception_matched": entry.get("fp_exception_matched", ""),
        "related_gt_flag_index": entry.get("related_gt_flag_index"),
        "transcript_reliability": entry["transcript_reliability"],
        "classifier_confidence": entry["classifier_confidence"],
        "classifier_model": classifier_model,
        "taxonomy_version": TAXONOMY_VERSION,
        "classified_at": now,
        "prompt_hash": cur_prompt_hash,
    }


def classify_all(client, model: str, work_items: List[WorkItem], file_state: dict, policy_text: str,
                  batch_size: int, workers: int, classifier_model: str, cur_prompt_hash: str,
                  logger: StageLogger) -> Dict[str, dict]:
    """Classifies in batches and writes each batch's results straight back
    into its per_file JSON(s) as soon as that batch completes — a crash
    mid-run only loses the batch in flight, not everything already done.
    Per-file writes are serialized by a lock keyed on file path, since two
    batches running concurrently can touch flags in the same file."""
    batches = [work_items[i:i + batch_size] for i in range(0, len(work_items), batch_size)]
    logger.info(f"classifying {len(work_items)} flag(s) in {len(batches)} batch(es) of up to {batch_size} via {model}")
    all_results: Dict[str, dict] = {}
    results_lock = threading.Lock()
    write_lock = threading.Lock()

    def process(b_idx, batch):
        ids = [it.item_id for it in batch]
        logger.info(f"[classify batch {b_idx}/{len(batches)}] {len(batch)} flag(s)")
        got = classify_batch(client, model, batch, policy_text, logger)
        missing = [it for it in batch if it.item_id not in got]
        if missing:
            logger.warn(f"[classify batch {b_idx}] {len(missing)} flag(s) missing/invalid — retrying individually")
            retried = classify_batch(client, model, missing, policy_text, logger)
            got.update(retried)
            for it in missing:
                if it.item_id not in got:
                    logger.warn(f"{it.item_id}: still invalid after retry — marking UNCLEAR")
                    got[it.item_id] = unclear_fallback("classifier returned invalid/missing output after retry")

        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        touched_files = set()
        with write_lock:
            for it in batch:
                entry = got.get(it.item_id) or unclear_fallback("no classifier output returned")
                state = file_state[(it.model_key, it.file_id)]
                apply_entry(state, it, entry, classifier_model, cur_prompt_hash, now)
                touched_files.add((it.model_key, it.file_id))
            for key in touched_files:
                state = file_state[key]
                state["path"].write_text(json.dumps(state["data"], ensure_ascii=False, indent=2))

        with results_lock:
            all_results.update(got)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, i, b) for i, b in enumerate(batches, 1)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"classify batch worker failed: {exc}\n{traceback.format_exc()}")

    return all_results


# ---------------------------------------------------------------------------
# Flat table + aggregates + adjusted precision + review export
# ---------------------------------------------------------------------------
FP_FLAGS_FIELDS = [
    "model", "language", "file_id", "file_bucket", "dataset_origin_label",
    "model_category", "timestamp", "excerpt", "translation", "model_confidence",
    "logprob_derived_confidence", "entropy_mean", "fp_type", "fp_reason",
    "evidence_transcript_quote", "fp_exception_matched", "related_gt_flag_index",
    "transcript_reliability", "classifier_confidence", "prompt_hash",
]


def gather_all_classified_rows(model_keys: List[str]) -> List[dict]:
    rows = []
    for model_key in model_keys:
        per_file_dir = ar.OUTPUT_DIR / model_key / "per_file"
        if not per_file_dir.is_dir():
            continue
        for cache_path in sorted(per_file_dir.glob("*.json")):
            data = json.loads(cache_path.read_text())
            for mf in data.get("model_flags", []):
                fpc = mf.get("fp_classification")
                if not fpc:
                    continue
                rows.append({
                    "model": model_key,
                    "language": data.get("language", ""),
                    "file_id": data.get("file_id", ""),
                    "file_bucket": data.get("file_bucket", ""),
                    "dataset_origin_label": mf.get("dataset_origin_label", ""),
                    "model_category": mf.get("category", ""),
                    "timestamp": mf.get("timestamp", ""),
                    "excerpt": mf.get("excerpt", ""),
                    "translation": mf.get("translation", ""),
                    "model_confidence": r2(mf.get("model_confidence")),
                    "logprob_derived_confidence": r2(mf.get("logprob_derived_confidence")),
                    "entropy_mean": r2(mf.get("entropy_mean")),
                    "fp_type": fpc.get("fp_type", ""),
                    "fp_reason": fpc.get("fp_reason", ""),
                    "evidence_transcript_quote": fpc.get("evidence_transcript_quote", ""),
                    "fp_exception_matched": fpc.get("fp_exception_matched", ""),
                    "related_gt_flag_index": fpc.get("related_gt_flag_index"),
                    "transcript_reliability": fpc.get("transcript_reliability", ""),
                    "classifier_confidence": fpc.get("classifier_confidence", ""),
                    "prompt_hash": fpc.get("prompt_hash", ""),
                })
    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def breakdown(rows: List[dict], group_keys: List[str]) -> List[dict]:
    by_group = defaultdict(list)
    for r in rows:
        by_group[tuple(r[k] for k in group_keys)].append(r)
    out = []
    for key, grp in sorted(by_group.items()):
        by_type = defaultdict(int)
        for r in grp:
            by_type[r["fp_type"]] += 1
        total = len(grp)
        for fp_type in FP_TYPES:
            n = by_type.get(fp_type, 0)
            if n == 0:
                continue
            row = dict(zip(group_keys, key))
            row["fp_type"] = fp_type
            row["n"] = n
            row["pct_of_fp"] = r2(100.0 * n / total) if total else None
            out.append(row)
    return out


def entropy_table(rows: List[dict]) -> List[dict]:
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["model"], r["fp_type"])].append(r)
    out = []
    for (model, fp_type), grp in sorted(by_key.items()):
        ent = [r["entropy_mean"] for r in grp if r["entropy_mean"] is not None]
        conf = [r["logprob_derived_confidence"] for r in grp if r["logprob_derived_confidence"] is not None]
        out.append({
            "model": model, "fp_type": fp_type, "n": len(grp),
            "mean_entropy": r2(sum(ent) / len(ent)) if ent else None,
            "mean_logprob_confidence": r2(sum(conf) / len(conf)) if conf else None,
        })
    return out


def adjusted_precision(model_keys: List[str]) -> List[dict]:
    """GT_MISS and MATCHER_MISS both reclassify as TP (in both cases the
    model was actually right — GT_MISS because ground truth never had the
    violation at all, MATCHER_MISS because ground truth DID but the scorer's
    matcher failed to pair it), and DUPLICATE_OR_EXTRA_INSTANCE is excluded
    from the FP count entirely (scoring artifact, not a real extra flag).
    Applied at both flag level (direct per-flag reclassification) and file
    level (an FP-bucket file becomes adjusted-TP if any of its flags
    reclassify as GT_MISS/MATCHER_MISS; TP/FN-bucket files are unaffected —
    they already have a real match or a real miss at flag level)."""
    out = []
    for model_key in model_keys:
        per_file_dir = ar.OUTPUT_DIR / model_key / "per_file"
        if not per_file_dir.is_dir():
            continue
        flag_tp = flag_fp = flag_fp_raw = flag_tp_raw = 0
        file_tp = file_fp = file_tp_raw = file_fp_raw = 0
        for cache_path in sorted(per_file_dir.glob("*.json")):
            data = json.loads(cache_path.read_text())
            flag_tp_raw += data.get("flag_tp", 0)
            flag_fp_raw += data.get("flag_fp", 0)
            tp_reclass_n = 0
            dup_n = 0
            for mf in data.get("model_flags", []):
                fpc = mf.get("fp_classification")
                if not fpc:
                    continue
                if fpc["fp_type"] in ("GT_MISS", "MATCHER_MISS"):
                    tp_reclass_n += 1
                elif fpc["fp_type"] == "DUPLICATE_OR_EXTRA_INSTANCE":
                    dup_n += 1
            flag_tp += data.get("flag_tp", 0) + tp_reclass_n
            flag_fp += data.get("flag_fp", 0) - tp_reclass_n - dup_n

            bucket = data.get("file_bucket")
            if bucket == "TP":
                file_tp_raw += 1
                file_tp += 1
            elif bucket == "FP":
                file_fp_raw += 1
                if tp_reclass_n > 0:
                    file_tp += 1
                else:
                    file_fp += 1

        raw_flag_p = r2(flag_tp_raw / (flag_tp_raw + flag_fp_raw)) if (flag_tp_raw + flag_fp_raw) else None
        adj_flag_p = r2(flag_tp / (flag_tp + flag_fp)) if (flag_tp + flag_fp) else None
        raw_file_p = r2(file_tp_raw / (file_tp_raw + file_fp_raw)) if (file_tp_raw + file_fp_raw) else None
        adj_file_p = r2(file_tp / (file_tp + file_fp)) if (file_tp + file_fp) else None

        out.append({
            "model": model_key,
            "flag_tp_raw": flag_tp_raw, "flag_fp_raw": flag_fp_raw, "precision_raw_flag": raw_flag_p,
            "flag_tp_adj": flag_tp, "flag_fp_adj": flag_fp, "precision_adjusted_flag": adj_flag_p,
            "file_tp_raw": file_tp_raw, "file_fp_raw": file_fp_raw, "precision_raw_file": raw_file_p,
            "file_tp_adj": file_tp, "file_fp_adj": file_fp, "precision_adjusted_file": adj_file_p,
        })
    return out


def write_reports(model_keys: List[str]):
    rows = gather_all_classified_rows(model_keys)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "fp_flags.csv", rows, FP_FLAGS_FIELDS)
    write_csv(OUTPUT_DIR / "fp_type_breakdown_overall.csv",
              breakdown(rows, ["model"]), ["model", "fp_type", "n", "pct_of_fp"])
    write_csv(OUTPUT_DIR / "fp_type_breakdown_by_language.csv",
              breakdown(rows, ["model", "language"]), ["model", "language", "fp_type", "n", "pct_of_fp"])
    write_csv(OUTPUT_DIR / "fp_type_breakdown_by_category.csv",
              breakdown(rows, ["model", "model_category"]), ["model", "model_category", "fp_type", "n", "pct_of_fp"])
    write_csv(OUTPUT_DIR / "fp_type_entropy.csv", entropy_table(rows),
              ["model", "fp_type", "n", "mean_entropy", "mean_logprob_confidence"])
    write_csv(OUTPUT_DIR / "adjusted_precision.csv", adjusted_precision(model_keys), [
        "model", "flag_tp_raw", "flag_fp_raw", "precision_raw_flag",
        "flag_tp_adj", "flag_fp_adj", "precision_adjusted_flag",
        "file_tp_raw", "file_fp_raw", "precision_raw_file",
        "file_tp_adj", "file_fp_adj", "precision_adjusted_file",
    ])
    write_review_sample(rows)
    return rows


def write_review_sample(rows: List[dict], seed: int = 42):
    selected = {}

    def key(r):
        return (r["model"], r["file_id"], r["timestamp"], r["model_category"])

    for r in rows:
        # GT_MISS and MATCHER_MISS both reclassify an FP as an adjusted TP —
        # equally consequential, equally worth a human second look. Observed
        # in practice: MATCHER_MISS can be over-applied on "same topic, not
        # actually same quote" grounds once the flag sees an unmatched GT
        # entry nearby, so it needs the same scrutiny as GT_MISS, not less.
        if r["fp_type"] in ("GT_MISS", "MATCHER_MISS"):
            selected[key(r)] = r
    for r in rows:
        if r["classifier_confidence"] == "low" or r["transcript_reliability"] == "suspect":
            selected[key(r)] = r

    rng = random.Random(seed)
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["model"], r["fp_type"])].append(r)
    for group_rows in by_group.values():
        sample = group_rows if len(group_rows) <= 5 else rng.sample(group_rows, 5)
        for r in sample:
            selected[key(r)] = r

    out_rows = list(selected.values())
    fieldnames = FP_FLAGS_FIELDS + ["human_fp_type", "human_notes"]
    for r in out_rows:
        r.setdefault("human_fp_type", "")
        r.setdefault("human_notes", "")
    write_csv(OUTPUT_DIR / "review_sample.csv", out_rows, fieldnames)


def score_review(path: str):
    p = Path(path)
    rows = list(csv.DictReader(open(p)))
    scored = [r for r in rows if r.get("human_fp_type")]
    if not scored:
        print(f"No rows with human_fp_type filled in yet in {path}")
        return

    agree = sum(1 for r in scored if r["human_fp_type"] == r["fp_type"])
    print(f"Overall agreement: {agree}/{len(scored)} ({r2(100*agree/len(scored))}%)")

    by_type = defaultdict(lambda: [0, 0])
    for r in scored:
        by_type[r["fp_type"]][1] += 1
        if r["human_fp_type"] == r["fp_type"]:
            by_type[r["fp_type"]][0] += 1
    print(f"\n{'fp_type':<30} {'agree/n':>10} {'pct':>8}")
    for fp_type in FP_TYPES:
        if fp_type not in by_type:
            continue
        a, n = by_type[fp_type]
        print(f"{fp_type:<30} {a:>4}/{n:<5} {r2(100*a/n):>7}%")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None, help="Comma-separated subset of model keys; default = all")
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated subset of languages; default = all")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of flags classified this run")
    parser.add_argument("--dry-run", action="store_true", help="Classify 3 flags, print requests/responses, write nothing")
    parser.add_argument("--classifier-model", type=str, default=config.GEMINI_CLASSIFIER_MODEL)
    parser.add_argument("--force", action="store_true", help="Reclassify even if a matching fp_classification is cached")
    parser.add_argument("--batch-size", type=int, default=5, help="Flags sent to the classifier per Gemini call")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--score-review", type=str, default=None, metavar="PATH",
                         help="Read back a filled-in review CSV and report classifier/human agreement")
    parser.add_argument("--dataset-root", type=str, default=None,
                         help="Alternate dataset root, e.g. Dostt_dev — see analyze_results.py --dataset-root; "
                              "must match whatever root was used for the analyze_results.py run being classified")
    parser.add_argument("--results-tag", type=str, default=None,
                         help="Must match the --results-tag used for the analyze_results.py run being classified")
    args = parser.parse_args()

    if args.score_review:
        score_review(args.score_review)
        return

    global OUTPUT_DIR
    ar.configure_dataset_root(args.dataset_root, args.results_tag)  # before any ar.MODEL_DIRS/ar.OUTPUT_DIR/ar.unique_records() use
    OUTPUT_DIR = ar.OUTPUT_DIR / "fp_classification"

    logger = StageLogger("classify_fps")
    model_keys = args.models.split(",") if args.models else list(ar.MODEL_DIRS)
    languages = args.languages.split(",") if args.languages else None

    cur_prompt_hash = prompt_hash()
    policy_text = prompt_loader.load_raw_prompt(config.CLASSIFICATION_PROMPT_PATH)
    logger.info(f"prompt_hash={cur_prompt_hash} classifier_model={args.classifier_model}")

    client = gemini_client.get_client()

    file_state, work_items = collect_work_items(
        model_keys, languages, args.force, cur_prompt_hash, args.classifier_model,
        client, args.batch_size, args.workers, logger,
    )
    logger.info(f"{len(work_items)} flag(s) need classification across {len(file_state)} file(s)")

    if args.limit:
        work_items = work_items[: args.limit]

    if args.dry_run:
        sample = work_items[:3]
        for it in sample:
            block = build_item_block(it, policy_text)
            print("\n" + "=" * 80)
            print(f"REQUEST for {it.item_id}")
            print("=" * 80)
            print(block)
            result = classify_batch(client, args.classifier_model, [it], policy_text, logger)
            print("-" * 80)
            print(f"RESPONSE for {it.item_id}")
            print("-" * 80)
            print(json.dumps(result.get(it.item_id, unclear_fallback("no output")), ensure_ascii=False, indent=2))
        logger.info("DRY RUN complete — nothing written.")
        return

    write_backfill(file_state)
    classify_all(client, args.classifier_model, work_items, file_state, policy_text,
                 args.batch_size, args.workers, args.classifier_model, cur_prompt_hash, logger)

    rows = write_reports(model_keys)
    logger.info(f"Classified {len(work_items)} flag(s); reports written to {OUTPUT_DIR}/")

    print(f"\nClassified {len(work_items)} flag(s) across {len(model_keys)} model(s). Reports in {OUTPUT_DIR}/")
    print(f"\n{'model':<30} {'fp_type':<30} {'n':>6} {'pct':>7}")
    by_model_rows = defaultdict(list)
    for r in rows:
        by_model_rows[r["model"]].append(r)
    for model_key in model_keys:
        grp = by_model_rows.get(model_key, [])
        total = len(grp)
        by_type = defaultdict(int)
        for r in grp:
            by_type[r["fp_type"]] += 1
        for fp_type in FP_TYPES:
            n = by_type.get(fp_type, 0)
            if n == 0:
                continue
            print(f"{model_key:<30} {fp_type:<30} {n:>6} {r2(100*n/total):>6}%")


if __name__ == "__main__":
    main()
