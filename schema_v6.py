"""v6 output schema for prompt_v6.py. Like schemas_v4.py, deliberately SEPARATE
from schemas.py: ground truth (stage2_classify.py / stage12_v2.py) must never
import this file.

Field order (t, seg, tr, f, spk, ctx, speech_act, j, violation, c) matches
prompt_v6.py's Per-Flag Steps exactly; model_json_schema() emits properties in
declaration order, and that schema is embedded in the prompt, so this order IS
the order the model is told to fill fields in.

Changes vs v4: adds spk (speaker) and ctx (surrounding words and the other
person's response) before speech_act; drops quote_type (models labelled
almost everything "verbatim", so it carried no signal). The "violation" key
name is unchanged so gemma_local.find_violation_token_index works as-is.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlagResultV6(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str = Field(alias="t", description="(string) Timestamp where the quote starts, MM:SS")
    excerpt: str = Field(alias="seg", description="(string) Exact quote as spoken, native language")
    translation: str = Field(alias="tr", description="(string) English translation of the quote")
    flag: str = Field(alias="f", description="(string) PlatformMove | SuspiciousActivity | Explicit-Flirting")
    speaker: Optional[Literal["expert", "user", "unclear"]] = Field(alias="spk", default=None, description="(string) Who said the quote")
    context: Optional[str] = Field(alias="ctx", default=None, description="(string) English: words just before and after the quote, and how the other person responded")
    speech_act: Optional[Literal["direct", "reported", "hypothetical", "denial"]] = Field(alias="speech_act", default=None)
    justification: str = Field(alias="j", description="(string) One-sentence justification")
    violation: Optional[Literal["yes", "no"]] = Field(alias="violation", default=None)
    confidence: Optional[float] = Field(alias="c", default=None, description="(float) Confidence the flag is a real violation")


class RawModelOutputV6(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    d: List[FlagResultV6] = Field(default_factory=list)


def raw_model_output_json_schema() -> dict:
    return RawModelOutputV6.model_json_schema(by_alias=True)