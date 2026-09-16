"""Text normalization + WER/CER/etc. metrics for comparing Indic-Conformer
output against the Gemini reference transcripts. Pure Python, no ML
dependencies — fully unit-testable on its own.

Normalization is applied IDENTICALLY to both sides of every comparison
(Gemini reference and Indic-Conformer hypothesis) specifically so that
formatting differences between the two pipelines (punctuation conventions,
literal "[inaudible]" markers, stray whitespace) never contribute to the
measured error rate — only genuine transcription differences should.
"""
import re
import unicodedata
from typing import Dict, List, Tuple

_INAUDIBLE_RE = re.compile(r"\[?\s*inaudible\s*\]?", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC-normalize, strip [inaudible] markers, strip all Unicode
    punctuation (category "P*"), collapse whitespace. Digits are
    deliberately preserved — phone numbers etc. matter for the
    policy-critical-term recall metric."""
    text = unicodedata.normalize("NFC", text)
    text = _INAUDIBLE_RE.sub(" ", text)
    text = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch
        for ch in text
    )
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _edit_ops(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Levenshtein alignment over token sequences (words or characters).
    Returns (substitutions, deletions, insertions, matches)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    i, j = n, m
    sub = dele = ins = match = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            match += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins, match


def word_error_rate(reference: str, hypothesis: str) -> Dict:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens:
        return {"wer": None, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_word_count": 0}
    sub, dele, ins, _ = _edit_ops(ref_tokens, hyp_tokens)
    wer = (sub + dele + ins) / len(ref_tokens)
    return {"wer": wer, "substitutions": sub, "deletions": dele, "insertions": ins, "ref_word_count": len(ref_tokens)}


def char_error_rate(reference: str, hypothesis: str) -> Dict:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return {"cer": None, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_char_count": 0}
    sub, dele, ins, _ = _edit_ops(ref_chars, hyp_chars)
    cer = (sub + dele + ins) / len(ref_chars)
    return {"cer": cer, "substitutions": sub, "deletions": dele, "insertions": ins, "ref_char_count": len(ref_chars)}


def policy_term_recall(excerpts: List[str], hypothesis: str, min_word_overlap: float = 0.7) -> Dict:
    """For each ground-truth excerpt (a cited policy-relevant quote), checks
    whether most of its words individually appear in the hypothesis text —
    a fuzzy containment check rather than exact substring match, since minor
    transcription differences shouldn't fail an otherwise-preserved excerpt.
    Returns per-excerpt hits and an overall recall fraction."""
    norm_hyp_words = set(normalize_text(hypothesis).split())
    results = []
    for excerpt in excerpts:
        excerpt_words = normalize_text(excerpt).split()
        if not excerpt_words:
            continue
        found = sum(1 for w in excerpt_words if w in norm_hyp_words)
        frac = found / len(excerpt_words)
        results.append({"excerpt": excerpt, "word_overlap_frac": frac, "preserved": frac >= min_word_overlap})
    n_preserved = sum(1 for r in results if r["preserved"])
    recall = n_preserved / len(results) if results else None
    return {"per_excerpt": results, "recall": recall, "n_excerpts": len(results), "n_preserved": n_preserved}


def boundary_corruption_rate(chunk_seconds: float, audio_duration_sec: float, reference_segment_starts: List[float]) -> Dict:
    """Approximates whether chunk boundaries fall mid-utterance. Gemini's
    segments carry only a start timestamp (no explicit end), so segment N's
    span is approximated as [start_N, start_{N+1}). A chunk boundary that
    lands strictly inside some segment's span (not within 0.5s of its own
    start) is assumed to have cut that utterance in half."""
    if not reference_segment_starts:
        return {"n_boundaries": 0, "n_corrupted": 0, "rate": None}

    starts = sorted(reference_segment_starts)
    spans = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
    if spans:
        spans.append((starts[-1], audio_duration_sec))

    n_boundaries, n_corrupted = 0, 0
    boundary = chunk_seconds
    while boundary < audio_duration_sec:
        n_boundaries += 1
        for span_start, span_end in spans:
            if span_start + 0.5 < boundary < span_end - 0.05:
                n_corrupted += 1
                break
        boundary += chunk_seconds

    rate = n_corrupted / n_boundaries if n_boundaries else None
    return {"n_boundaries": n_boundaries, "n_corrupted": n_corrupted, "rate": rate}
