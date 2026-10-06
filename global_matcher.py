"""Global one-to-one GT<->model flag matcher.

Replaces the old pairwise approach in analyze_results.py (deterministic
same-script text-similarity pre-pass + an LLM leftover call that only ever
saw {category, translation} for each side) after an audit of 28 changed
pairs under a prompt patch found the old design wasn't converging — its
failures all traced back to the same root causes, not independent bugs:

  - It considered pairs ad hoc rather than picking each ground-truth flag's
    SINGLE BEST candidate among everything available — "better candidate
    missed" was the single largest failure category in the audit.
  - It scored native-text similarity even when the model's own transcription
    was in the WRONG SCRIPT entirely (a real, frequent failure mode — see
    script_match_fraction below) — a false negative on every such pair.
  - It had no defense against a model timestamp defaulting to its chunk's
    start (e.g. "00:00" for the first flag of the first chunk) — the old
    matcher (and the audit) saw these as "17 seconds away" when the true
    position was unknown, not exactly 00:00.
  - The leftover LLM step never saw native text or timestamps at all, only
    translations — so it had no way to notice a timestamp was implausible
    or that two translations were loose paraphrases of the same instant.

This module fixes all four in one pass:
  1. score_pair() scores EVERY (GT, model) pair within a wide time window on
     translation-embedding similarity, native-text similarity (skipped, not
     penalized, when either side's script doesn't match the expected
     language), time proximity (with chunk-boundary timestamps treated as
     "unknown position within the chunk" rather than an exact point), and a
     category match bonus/mismatch penalty (not a hard filter — cross-
     category pairs stay possible, so wrong-category cases still surface).
  2. assign_pairs_for_file() runs the Hungarial algorithm (scipy's
     linear_sum_assignment) over the full score matrix, so every ground-
     truth flag gets its single best available candidate, not just
     whichever candidate happened to be considered first.
  3. llm_verify_pairs() asks an LLM to confirm or reject each assigned pair,
     giving it native text, translation, AND timestamps for both sides
     (unlike the old leftover-matching prompt).
  4. iterative_match_file() re-runs the assignment on anything rejected, so
     a ground-truth flag isn't left unmatched just because its first-best
     candidate turned out wrong — it gets its next-best candidate instead.

Every assigned pair's match_detail records its score, score components,
whether the LLM confirmed it, and whether either side's timestamp was
treated as "unknown" — so a future audit is a cache lookup, not a multi-hour
investigation like the one that led here.
"""
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import gemini_client

# Same chunking default as gemma_local.py's DEFAULT_CHUNK_SECONDS — not
# threaded through per-file from the actual run (the per-file cache doesn't
# retain it per flag), so this is a known simplifying assumption: every
# Dostt_dev/Dostt run in this project used the default 28s chunking unless
# explicitly overridden, which none of v1-v6 were.
ASSUMED_CHUNK_SECONDS = 28.0
CHUNK_BOUNDARY_TOLERANCE_SEC = 0.5

MAX_TIME_WINDOW_SEC = 60.0
TIME_DECAY_TAU_SEC = 20.0  # score(0s)=1.0, score(20s)=0.37, score(60s)=0.05
UNKNOWN_TIME_SCORE = 0.7  # neutral-ish score when the model's timestamp is a chunk boundary (true position unknown)

W_TRANSLATION = 0.45
W_NATIVE = 0.25
W_TIME = 0.30
CATEGORY_ADJUST = 0.15

# Hard gate (see score_pair): below this, a pair is excluded outright,
# regardless of how close in time or how category-matched it is. "Moderate"
# rather than high, since translation phrasing legitimately varies.
MIN_CONTENT_SIMILARITY = 0.5

MIN_SCORE_TO_ASSIGN = 0.45
DISALLOWED_COST = 1e6  # effectively excludes a pair from the Hungarian assignment without requiring -inf

# Unicode block per language — used only to detect whether a flag's native
# excerpt is even in the expected script before trusting native-text
# similarity; this is NOT a correctness check on the transcription itself.
SCRIPT_RANGES = {
    "hindi": [(0x0900, 0x097F)],       # Devanagari
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
}


def script_match_fraction(text: str, language: str) -> Optional[float]:
    """Fraction of this text's ALPHABETIC characters that fall in the
    expected Unicode script for `language`. Returns None if there are no
    alphabetic characters to judge (e.g. an all-digit/punctuation excerpt —
    a phone number read out in digits has no script to be wrong about)."""
    ranges = SCRIPT_RANGES.get(language)
    if not ranges:
        return None
    alphabetic = [c for c in text if c.isalpha()]
    if not alphabetic:
        return None
    in_script = sum(1 for c in alphabetic if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return in_script / len(alphabetic)


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


def is_chunk_boundary_timestamp(ts_sec: float) -> bool:
    """True if ts_sec sits within tolerance of a multiple of
    ASSUMED_CHUNK_SECONDS — the signature of a model defaulting to its
    chunk's start time rather than reporting a genuine within-chunk
    position (see module docstring)."""
    remainder = ts_sec % ASSUMED_CHUNK_SECONDS
    return remainder <= CHUNK_BOUNDARY_TOLERANCE_SEC or (ASSUMED_CHUNK_SECONDS - remainder) <= CHUNK_BOUNDARY_TOLERANCE_SEC


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_texts(client, texts: List[str], model: str = "text-multilingual-embedding-002") -> List[np.ndarray]:
    """One batched call per file's worth of texts (GT + model translations
    together) — typically well under 100 texts per file.

    The embedding API silently drops empty/whitespace-only inputs instead
    of returning a zero vector for them, which breaks the 1:1 index
    alignment callers rely on — substitute a placeholder so every input
    always gets exactly one embedding back."""
    if not texts:
        return []
    safe_texts = [t if (t and t.strip()) else "(empty)" for t in texts]
    result = client.models.embed_content(model=model, contents=safe_texts)
    embeddings = [np.array(e.values) for e in result.embeddings]
    if len(embeddings) != len(safe_texts):
        raise RuntimeError(
            f"embed_content returned {len(embeddings)} embeddings for {len(safe_texts)} inputs — "
            f"index alignment can't be trusted, refusing to guess which ones are missing."
        )
    return embeddings


def score_pair(gt: dict, model: dict, language: str,
               gt_emb: np.ndarray, model_emb: np.ndarray) -> Optional[dict]:
    """Returns None if the pair is outside the time window entirely
    (never assignable); otherwise a dict with the combined score and its
    components, for both the assignment step and the audit trail."""
    gt_ts = _parse_ts_sec(gt.get("timestamp", ""))
    model_ts = _parse_ts_sec(model.get("timestamp", ""))
    if gt_ts is None or model_ts is None:
        return None

    model_ts_unknown = is_chunk_boundary_timestamp(model_ts)
    if model_ts_unknown:
        time_component = UNKNOWN_TIME_SCORE
        in_window = True  # an unknown-position timestamp can't be used to EXCLUDE a pair
    else:
        delta = abs(gt_ts - model_ts)
        if delta > MAX_TIME_WINDOW_SEC:
            return None
        in_window = True
        time_component = float(np.exp(-delta / TIME_DECAY_TAU_SEC))
    if not in_window:
        return None

    translation_sim = (cosine_sim(gt_emb, model_emb) + 1) / 2  # rescale [-1,1] -> [0,1]

    gt_script_ok = script_match_fraction(gt.get("excerpt", ""), language)
    model_script_ok = script_match_fraction(model.get("excerpt", ""), language)
    native_available = (
        gt_script_ok is not None and gt_script_ok >= 0.5
        and model_script_ok is not None and model_script_ok >= 0.5
    )
    native_sim = _native_text_similarity(gt.get("excerpt", ""), model.get("excerpt", "")) if native_available else None

    # content_sim = whichever signal is stronger, not a fixed blend: strong
    # evidence from EITHER text should be enough on its own, and a weak or
    # unreliable native score (or one skipped entirely because the script
    # didn't match) should never drag a pair down. Native only ever helps
    # here, never hurts, because it's only considered when the script check
    # already passed.
    content_sim = max(translation_sim, native_sim) if native_sim is not None else translation_sim

    # Hard gate: a pair needs at least moderate content similarity before
    # time proximity and category get to count at all — otherwise "close in
    # time, same category" alone could pass a pair that isn't the same
    # utterance (e.g. two unrelated lines a few seconds apart). This encodes
    # what "same utterance" actually means, independent of any specific case.
    if content_sim < MIN_CONTENT_SIMILARITY:
        return None

    combined = (W_TRANSLATION + W_NATIVE) * content_sim + W_TIME * time_component

    same_category = gt.get("category") == model.get("category")
    combined += CATEGORY_ADJUST if same_category else -CATEGORY_ADJUST
    combined = max(0.0, min(1.0, combined))

    return {
        "score": combined,
        "content_sim": round(content_sim, 3),
        "translation_sim": round(translation_sim, 3),
        "native_sim": round(native_sim, 3) if native_sim is not None else None,
        "time_component": round(time_component, 3),
        "same_category": same_category,
        "model_ts_unknown": model_ts_unknown,
    }


def _native_text_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def assign_pairs_for_file(gt_flags: List[dict], model_flags: List[dict], language: str,
                           gt_embs: List[np.ndarray], model_embs: List[np.ndarray]) -> List[Tuple[int, int, dict]]:
    """Optimal one-to-one assignment (Hungarian algorithm) over every
    in-window (GT, model) pair's score — each GT flag gets its single best
    available candidate, not just the first one considered. Pairs scoring
    below MIN_SCORE_TO_ASSIGN are dropped after assignment (a forced
    low-score pairing from an unbalanced matrix is not a real match)."""
    n_gt, n_model = len(gt_flags), len(model_flags)
    if n_gt == 0 or n_model == 0:
        return []

    cost = np.full((n_gt, n_model), DISALLOWED_COST)
    detail = {}
    for gi in range(n_gt):
        for mi in range(n_model):
            d = score_pair(gt_flags[gi], model_flags[mi], language, gt_embs[gi], model_embs[mi])
            if d is not None:
                cost[gi, mi] = 1.0 - d["score"]
                detail[(gi, mi)] = d

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for gi, mi in zip(row_ind, col_ind):
        d = detail.get((gi, mi))
        if d is not None and d["score"] >= MIN_SCORE_TO_ASSIGN:
            pairs.append((int(gi), int(mi), d))
    return pairs


LLM_VERIFY_INSTRUCTION = """
You are verifying candidate matches between a ground-truth policy-violation
flag and a candidate model's flag for the SAME audio call. Each candidate
pair was proposed by an automated scorer using translation similarity, time
proximity, and category — your job is to catch cases the scorer got wrong.

For each pair, you are given: category, timestamp, native-language text, and
English translation for BOTH the GT flag and the MODEL flag.

IMPORTANT: the MODEL side's native text is the model's OWN transcription,
which is frequently unreliable — sometimes garbled, or written in the WRONG
SCRIPT entirely. Never reject a pair just because the native text looks
unrelated or is in an unexpected script — judge primarily on whether the
translations describe the SAME specific quote/moment (not just the same
general topic or category) and whether the timestamps are plausible for
that. A MODEL timestamp marked "(uncertain, chunk start)" means its true
position within that time window is unknown — do not reject a pair for a
time gap caused only by that uncertainty.

Don't reject a pair solely because of question-vs-statement phrasing (e.g.
"Will you show for 200?" vs "I'll show for 200") — translation often blurs
this distinction in these languages, and a question and its answer are
legitimately two different utterances, often by two different speakers, for
the SAME moment in the conversation. Judge whether it's the same moment and
the same content, not the grammatical mood. Given close timestamps and a
plausible speaker relationship, matching either side of such a question/
answer exchange to the ground-truth flag is acceptable, since ground truth
usually records only one flag for that moment.

Confirm a pair only if you're confident the GT and MODEL items describe the
SAME specific incident. Reject if they're clearly different moments or
different content — a different specific number, amount, or named detail is
a real difference, not a phrasing difference, and should still be rejected.

Return ONE raw, minified JSON object:
{"verdicts": [{"pair_index": 0, "confirmed": true, "reason": "..."}, ...]}
Include EVERY pair_index given, in any order. Your entire response must be
only the minified JSON object and nothing else.
""".strip()

LLM_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_index": {"type": "integer"},
                    "confirmed": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["pair_index", "confirmed"],
            },
        },
    },
    "required": ["verdicts"],
}


def llm_verify_pairs(client, match_model: str, gt_flags: List[dict], model_flags: List[dict],
                      pairs: List[Tuple[int, int, dict]], logger,
                      decision_log: Optional[list] = None) -> Tuple[List[int], List[int]]:
    """Returns (confirmed_pair_positions, rejected_pair_positions) — indices
    into `pairs`, not into gt_flags/model_flags. If decision_log is given,
    appends one record per pair (score, confirmed, reason, both
    translations) for later diagnosis of verifier behavior by score band."""
    if not pairs:
        return [], []

    lines = []
    for idx, (gi, mi, d) in enumerate(pairs):
        gt, m = gt_flags[gi], model_flags[mi]
        m_ts_label = f'{m.get("timestamp","")} (uncertain, chunk start)' if d["model_ts_unknown"] else m.get("timestamp", "")
        lines.append(
            f'[PAIR {idx}]\n'
            f'  GT:    category={gt["category"]} timestamp={gt.get("timestamp","")} native="{gt.get("excerpt","")}" translation="{gt["translation"]}"\n'
            f'  MODEL: category={m["category"]} timestamp={m_ts_label} native="{m.get("excerpt","")}" translation="{m["translation"]}"'
        )
    prompt_content = f"{LLM_VERIFY_INSTRUCTION}\n\n" + "\n\n".join(lines)

    def do_call():
        return gemini_client.generate_text(client, match_model, contents=[prompt_content], response_json_schema=LLM_VERIFY_SCHEMA)

    def on_retry(attempt, max_retries, delay, exc):
        logger.warn(f"verify batch of {len(pairs)} pair(s): attempt {attempt}/{max_retries} failed ({exc}); retrying in {delay:.1f}s")

    try:
        raw_text = gemini_client.call_with_retries(do_call, on_retry=on_retry)
        parsed = gemini_client.parse_json_lenient(raw_text)
    except Exception as exc:
        logger.error(f"verify batch of {len(pairs)} pair(s) failed entirely: {exc} — treating all as unconfirmed")
        return [], list(range(len(pairs)))

    confirmed, rejected = [], []
    verdicts_by_verbose = {v.get("pair_index"): v for v in parsed.get("verdicts", [])}
    for idx in range(len(pairs)):
        v = verdicts_by_verbose.get(idx, {})
        is_confirmed = bool(v.get("confirmed"))
        if is_confirmed:
            confirmed.append(idx)
        else:
            rejected.append(idx)
        if decision_log is not None:
            gi, mi, d = pairs[idx]
            decision_log.append({
                "score": d["score"], "confirmed": is_confirmed, "reason": v.get("reason", ""),
                "gt_translation": gt_flags[gi]["translation"], "model_translation": model_flags[mi]["translation"],
            })
    return confirmed, rejected


def iterative_match_file(client, match_model: str, gt_flags: List[dict], model_flags: List[dict],
                          language: str, logger, max_rounds: int = 3,
                          decision_log: Optional[list] = None) -> List[Tuple[int, int, dict]]:
    """Runs assign -> LLM-verify -> remove-rejected-pair -> reassign, up to
    max_rounds times, so a GT flag that loses its first-best candidate to a
    rejection still gets a chance at its next-best one."""
    all_texts = [f["translation"] for f in gt_flags] + [f["translation"] for f in model_flags]
    all_embs = embed_texts(client, all_texts) if all_texts else []
    gt_embs = all_embs[:len(gt_flags)]
    model_embs = all_embs[len(gt_flags):len(gt_flags) + len(model_flags)]

    available_gt = set(range(len(gt_flags)))
    available_model = set(range(len(model_flags)))
    forbidden_pairs = set()  # specific (gi, mi) combos an LLM has already rejected
    confirmed_final: List[Tuple[int, int, dict]] = []

    for _round in range(max_rounds):
        if not available_gt or not available_model:
            break
        gi_list = sorted(available_gt)
        mi_list = sorted(available_model)
        sub_gt = [gt_flags[i] for i in gi_list]
        sub_model = [model_flags[i] for i in mi_list]
        sub_gt_embs = [gt_embs[i] for i in gi_list]
        sub_model_embs = [model_embs[i] for i in mi_list]

        sub_pairs = assign_pairs_for_file(sub_gt, sub_model, language, sub_gt_embs, sub_model_embs)
        # map back to real indices, dropping anything already forbidden
        real_pairs = []
        for sub_gi, sub_mi, d in sub_pairs:
            gi, mi = gi_list[sub_gi], mi_list[sub_mi]
            if (gi, mi) in forbidden_pairs:
                continue
            real_pairs.append((gi, mi, d))
        if not real_pairs:
            break

        confirmed_pos, rejected_pos = llm_verify_pairs(client, match_model, gt_flags, model_flags, real_pairs, logger, decision_log)
        for pos in confirmed_pos:
            gi, mi, d = real_pairs[pos]
            d = {**d, "llm_confirmed": True}
            confirmed_final.append((gi, mi, d))
            available_gt.discard(gi)
            available_model.discard(mi)
        if not rejected_pos:
            break
        for pos in rejected_pos:
            gi, mi, _d = real_pairs[pos]
            forbidden_pairs.add((gi, mi))

    return confirmed_final


def resolve_matches_global(client, match_model: str, batch_items: List[dict],
                            logger) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, List[dict]]]:
    """Drop-in-shaped replacement for analyze_results.resolve_matches's
    OUTPUT (file_id -> matched pairs), plus a second dict of per-file match
    detail (score/components/llm_confirmed/timestamp_unknown per pair) for
    the audit trail. One file at a time (matches the Hungarian assignment's
    per-file scope) — call sites should parallelize across files, not
    batch multiple files into one call like the old leftover matcher did."""
    pairs_by_file: Dict[str, List[Tuple[int, int]]] = {}
    detail_by_file: Dict[str, List[dict]] = {}
    for item in batch_items:
        file_id = item["file_id"]
        gt_flags, model_flags, language = item["gt_flags"], item["model_flags"], item.get("language", "")
        if not gt_flags or not model_flags:
            pairs_by_file[file_id] = []
            detail_by_file[file_id] = []
            continue
        matched = iterative_match_file(client, match_model, gt_flags, model_flags, language, logger)
        pairs_by_file[file_id] = [(gi, mi) for gi, mi, _d in matched]
        detail_by_file[file_id] = [
            {"gt_index": gi, "model_index": mi, **d} for gi, mi, d in matched
        ]
    return pairs_by_file, detail_by_file
