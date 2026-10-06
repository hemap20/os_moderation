"""v6 output schema — adds spk/ctx fields on top of v4's speech_act/violation
(drops v4's quote_type, which v6's prompt never asks for) for the
prompt_v6.py experiment. Deliberately SEPARATE from schemas.py/schemas_v4.py:
ground truth (stage2_classify.py / stage12_v2.py) always uses schemas.py +
the original prompt.py, regardless of what candidate-model experiment is
running — this file must never be imported by either of those two scripts.
See prompt_loader.schema_module_for_prompt() for the prompt->schema mapping
that keeps this automatic instead of relying on every caller remembering.

Field order below (t, seg, tr, f, spk, ctx, speech_act, j, violation, c)
matches prompt_v6.py's own numbered "Per-Flag Steps" list exactly —
pydantic's model_json_schema(by_alias=True) emits "properties" in
field-declaration order, and that JSON schema is what gets embedded in the
prompt via {json_schema_str}, so the declaration order here IS the order
the model is told to fill fields in.

All fields besides the identifying ones are Optional with default=None —
never required — for the same reason as schemas_v4.py: real model output is
unreliable about including every field, and a schema that raises on a
missing field would crash exactly the runs this is meant to measure.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlagResultV6(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str = Field(alias="t", description="(string) Timestamp of the flag")
    excerpt: str = Field(alias="seg", description="(string) Exact cited quote, native language, that triggered the flag")
    translation: str = Field(alias="tr", description="(string) Translation of the flagged content")
    flag: str = Field(alias="f", description="(string) The category of the flag")
    speaker: Optional[Literal["expert", "user", "unclear"]] = Field(alias="spk", default=None)
    context: Optional[str] = Field(alias="ctx", default=None, description="(string) Words just before/after the quote and how the other person responded")
    speech_act: Optional[Literal["direct", "reported", "hypothetical", "denial"]] = Field(alias="speech_act", default=None)
    justification: str = Field(alias="j", description="(string) Justification for why the flag was raised")
    violation: Optional[Literal["yes", "no"]] = Field(alias="violation", default=None)
    confidence: Optional[float] = Field(alias="c", default=None, description="(Float) Confidence score for the flag")


class RawModelOutputV6(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    d: List[FlagResultV6] = Field(default_factory=list)


def raw_model_output_json_schema() -> dict:
    """The schema to substitute into prompt_v6.py's {json_schema_str} — same
    role as schemas.raw_model_output_json_schema(), just for the v6 field
    set. by_alias=True so the model is told to emit the short keys."""
    return RawModelOutputV6.model_json_schema(by_alias=True)
