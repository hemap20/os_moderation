"""Combined transcript+classification+determination schema for the Dostt
(v2) pipeline. One JSON per file, as approved: transcript first,
classification appended, then the FN<->TN / TP/FP-mismatch determination.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from schemas import GroundTruthFlag, TranscriptSegment


class TranscriptBlock(BaseModel):
    segments: List[TranscriptSegment] = Field(default_factory=list)
    full_text: str = ""
    audio_duration_sec: Optional[float] = None
    transcript_end_sec: Optional[float] = None
    gap_sec: Optional[float] = None
    incomplete_transcript: bool = False
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None


class ClassificationBlock(BaseModel):
    ground_truth_flags: List[GroundTruthFlag] = Field(default_factory=list)
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None


class Determination(BaseModel):
    final_bucket: Literal["TP", "FP", "FN", "TN"]
    final_category: Optional[str] = None
    reassigned: bool = False
    reassignment_note: Optional[str] = None
    needs_review: bool = False
    review_reason: Optional[str] = None


class CombinedResult(BaseModel):
    file_id: str
    source_path: str
    language: str
    original_bucket: Literal["TP", "FP", "FN"]
    original_category: Optional[str] = None  # full category name, None for FN

    transcript: TranscriptBlock
    classification: ClassificationBlock
    determination: Optional[Determination] = None

    status: Literal["success", "error"] = "success"
    error: Optional[str] = None
