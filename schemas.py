"""Pydantic schemas shared across stages.

IMPORTANT: prompt.py (repo root) is ONE shared classification prompt/schema,
reused for both Stage 2 (Gemini, fed the Stage 1 transcript as text) and
Stage 3 (Gemma, fed the raw audio). Both therefore produce the same RAW wire
schema (RawModelOutput/FlagResult below) — given verbatim by the user:

    {"d": [{"f": "PlatformMove", "t": "07:33", "seg": "<native quote>",
            "tr": "<english translation>", "j": "<justification>", "c": 0.9}]}

Note: there is no "vb" (verbatim/paraphrase) output field in this version —
the prompt still does that check internally (see prompt.py step 3) to decide
confidence, it just isn't surfaced as a field.

Ground truth (Stage 2) is persisted with every field renamed to a
``ground_truth_``-prefixed name (see raw_to_ground_truth() below) so it can
never be confused with Gemma's (Stage 3) raw output fields when the two are
compared side by side in Stage 4. Stage 3 keeps the raw field names as-is.
"""
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")


def _normalize_timestamp(raw_ts: str) -> str:
    """Strip stray formatting (e.g. brackets) the model may echo back from
    the input transcript's "[MM:SS] text" line format, keeping just MM:SS."""
    match = _TIMESTAMP_RE.search(raw_ts)
    return match.group(0) if match else raw_ts


# ---------------------------------------------------------------------------
# Stage 1 — transcription. Single-point timestamp per line ("t").
# ---------------------------------------------------------------------------
class TranscriptSegment(BaseModel):
    t: str = Field(..., description="Timestamp of this line, e.g. '00:12'")
    text: str = Field(..., description="Native-language transcript text for this line")


class Transcript(BaseModel):
    file_id: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    full_text: str = ""
    audio_duration_sec: Optional[float] = None
    transcript_end_sec: Optional[float] = None
    gap_sec: Optional[float] = None
    incomplete_transcript: bool = False
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Shared raw output schema — user's exact production model. populate_by_name
# lets us construct it from either the alias (f/t/tr/vb/c, what the model
# actually returns) or the long field name; by_alias=True on
# model_json_schema() is what gets sent to Gemini/Gemma as {json_schema_str}
# so the model is told to emit the short alias keys.
# ---------------------------------------------------------------------------
class FlagResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    flag: str = Field(alias="f", description="(string) The category of the flag")
    timestamp: str = Field(alias="t", description="(string) Timestamp of the flag")
    excerpt: str = Field(alias="seg", description="(string) Exact cited quote, native language, that triggered the flag")
    translation: str = Field(alias="tr", description="(string) Translation of the flagged content")
    justification: str = Field(alias="j", description="(string) Justification for why the flag was raised")
    confidence: float = Field(alias="c", description="(Float) Confidence score for the flag")


class RawModelOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    d: List[FlagResult] = Field(default_factory=list)


# Stage 3 (Gemma) keeps the raw shape as its canonical output type.
RawFlag = FlagResult
GemmaFlag = FlagResult
GemmaOutput = RawModelOutput


# ---------------------------------------------------------------------------
# Stage 2 — ground truth (Gemini's RawModelOutput, persisted with
# ground_truth_-prefixed field names for zero ambiguity vs. Gemma's output).
# ---------------------------------------------------------------------------
class GroundTruthFlag(BaseModel):
    ground_truth_category: str
    ground_truth_timestamp: str
    ground_truth_excerpt: str
    ground_truth_translation: str
    ground_truth_justification: str
    ground_truth_confidence: float


class GroundTruthClassification(BaseModel):
    file_id: str
    ground_truth_flags: List[GroundTruthFlag] = Field(default_factory=list)
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None


def raw_to_ground_truth(file_id: str, raw: RawModelOutput) -> GroundTruthClassification:
    flags = [
        GroundTruthFlag(
            ground_truth_category=flag.flag,
            ground_truth_timestamp=_normalize_timestamp(flag.timestamp),
            ground_truth_excerpt=flag.excerpt,
            ground_truth_translation=flag.translation,
            ground_truth_justification=flag.justification,
            ground_truth_confidence=flag.confidence,
        )
        for flag in raw.d
    ]
    return GroundTruthClassification(file_id=file_id, ground_truth_flags=flags)


def raw_model_output_json_schema() -> dict:
    """The schema to substitute into prompt.py's {json_schema_str} — used by
    BOTH Stage 2 (Gemini) and Stage 3 (Gemma) calls, since they share the
    same prompt and output contract. by_alias=True so the model is told to
    emit the short keys (f/t/tr/vb/c), matching FlagResult's real wire
    format."""
    return RawModelOutput.model_json_schema(by_alias=True)


# Back-compat aliases.
def ground_truth_json_schema() -> dict:
    return raw_model_output_json_schema()


def gemma_json_schema() -> dict:
    return raw_model_output_json_schema()
