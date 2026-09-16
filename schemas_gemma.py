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
