"""Output schema for locally-run Gemma 4 (E2B/E4B/12B) results.

Deliberately uses a ``model_``-prefixed field naming (not ``ground_truth_``)
per your explicit choice — Gemma's output lives in its own per-model,
per-thinking-mode directory tree (gemma_results/{model}_{thinking|nothinking}/),
never merged into the same file as real ground truth, but kept visually
distinct anyway.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TokenEntropy(BaseModel):
    per_token: List[float] = Field(default_factory=list)
    mean: Optional[float] = None


class TopKEntry(BaseModel):
    token: str
    logprob: float


class GemmaChunkFlag(BaseModel):
    model_category: str
    model_timestamp: str  # file-relative, chunk-offset-corrected
    model_excerpt: str
    model_translation: str
    model_justification: str
    model_confidence: Optional[float] = None  # self-reported "c" field

    # Logprob-derived signals — see gemma_local.py's
    # compute_flag_confidence_and_entropy() for exactly how these are
    # located against the raw token stream.
    logprob_decision: Optional[float] = None
    logprob_category: Optional[float] = None
    logprob_derived_confidence: Optional[float] = None
    excerpt_token_entropy: Optional[TokenEntropy] = None

    # --- v4-only fields (prompt_v4.py's speech_act/quote_type/violation) ---
    # Optional so pre-v4 results (and v4 results where the model omitted a
    # field — see the missing-field-count tracking in gemma_local.py) still
    # load fine; None here means "the model didn't emit this," never
    # imputed to any other value.
    model_speech_act: Optional[str] = None  # "direct" | "reported" | "hypothetical" | "denial"
    model_quote_type: Optional[str] = None  # "verbatim" | "paraphrase"
    model_violation: Optional[str] = None  # "yes" | "no", the model's own raw string

    # logprob of the model's chosen first token of the "violation" value,
    # located the same way logprob_category locates the category span (see
    # gemma_local.py's find_violation_token_index).
    logprob_violation: Optional[float] = None
    # P(violation == "yes"), however it had to be derived — see
    # p_violation_method for which of the three paths below was used:
    #   "direct":    violation == "yes" -> exp(logprob_violation) directly.
    #   "topk_yes":  violation == "no" but a "yes"-equivalent token was
    #                found in the top-K alternatives at that position ->
    #                exp(that alternative's logprob).
    #   "complement": violation == "no" and no "yes" alternative was in the
    #                 top-K -> 1 - exp(logprob_violation), a lower-bound
    #                 approximation (there may be "yes"-mass outside top-K).
    p_violation_yes: Optional[float] = None
    p_violation_method: Optional[str] = None
    # Top-K alternatives at ONLY the violation-token position (not every
    # token — that would bloat file size for no benefit elsewhere).
    violation_token_topk: Optional[List[TopKEntry]] = None


class GemmaFileResult(BaseModel):
    file_id: str
    model: str  # "e2b" / "e4b" / "12b"
    thinking: bool
    chunk_seconds: float
    flags: List[GemmaChunkFlag] = Field(default_factory=list)
    chunks_total: int = 0
    chunks_failed: int = 0
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None
    # Per-field missing-value counts across every flag in this file — e.g.
    # {"c": 2, "speech_act": 0, ...}. Only fields that were ever looked up
    # via .get() during parsing appear here; a field absent from this dict
    # (rather than present with 0) means this run's schema didn't ask for
    # it at all (e.g. speech_act under prompt.py/v2/v3).
    missing_field_counts: dict = Field(default_factory=dict)
