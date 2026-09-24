"""v4 output schema — adds speech_act/quote_type/violation fields for the
prompt_v4.py experiment. Deliberately SEPARATE from schemas.py: ground
truth (stage2_classify.py / stage12_v2.py) always uses schemas.py + the
original prompt.py, regardless of what candidate-model experiment is
running — this file must never be imported by either of those two scripts.
See prompt_loader.schema_module_for_prompt() for the prompt->schema mapping
that keeps this automatic instead of relying on every caller remembering.

Field order below (t, seg, tr, f, speech_act, quote_type, j, violation, c)
matches prompt_v4.py's own numbered field list exactly — pydantic's
model_json_schema(by_alias=True) emits "properties" in field-declaration
order, and that JSON schema is what gets embedded in the prompt via
{json_schema_str}, so the declaration order here IS the order the model is
told to fill fields in.

All fields besides the identifying ones are Optional with default=None —
never required — because the whole point of tracking model_confidence's
~10% missing rate (see analyze_results.py's ranking-metrics docstring) is
that real model output is unreliable about including every field; a
schema that raises on a missing field would crash exactly the runs this is
meant to measure.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlagResultV4(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str = Field(alias="t", description="(string) Timestamp of the flag")
    excerpt: str = Field(alias="seg", description="(string) Exact cited quote, native language, that triggered the flag")
    translation: str = Field(alias="tr", description="(string) Translation of the flagged content")
    flag: str = Field(alias="f", description="(string) The category of the flag")
    speech_act: Optional[Literal["direct", "reported", "hypothetical", "denial"]] = Field(alias="speech_act", default=None)
    quote_type: Optional[Literal["verbatim", "paraphrase"]] = Field(alias="quote_type", default=None)
    justification: str = Field(alias="j", description="(string) Justification for why the flag was raised")
    violation: Optional[Literal["yes", "no"]] = Field(alias="violation", default=None)
    confidence: Optional[float] = Field(alias="c", default=None, description="(Float) Confidence score for the flag")


class RawModelOutputV4(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    d: List[FlagResultV4] = Field(default_factory=list)


def raw_model_output_json_schema() -> dict:
    """The schema to substitute into prompt_v4.py's {json_schema_str} — same
    role as schemas.raw_model_output_json_schema(), just for the v4 field
    set. by_alias=True so the model is told to emit the short keys."""
    return RawModelOutputV4.model_json_schema(by_alias=True)
