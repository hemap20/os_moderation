"""Confidence scoring modes — read-only transformations layered on top of
already-matched, already-scored flags. A scoring mode only changes which
confidence value is used for thresholding, or removes a flag from
consideration entirely; it NEVER touches matching, ground truth, or the raw
model output. model_confidence itself is never modified anywhere in this
module — every function here is a pure read, callers store the result
under a separate key ("scored" in the per-file cache).

Field availability (whether a prompt version's schema even HAS speech_act,
spk, or violation at all) is determined via prompt_loader against that
version's actual prompt/schema files — never by inspecting whether a given
run's data happens to have the field populated. This is what makes the
difference between "the field doesn't exist for this version" (the rule
never fires) and "the field exists but the model left it out" (the rule
fires, since v6+'s own prompt treats a missing speech_act as suspicious).
"""
from pathlib import Path
from typing import Optional, Set

import prompt_loader

SCORING_MODES_VERSION = "sm_v1"  # bump this if any rule below changes

VALID_CATEGORIES = {"PlatformMove", "SuspiciousActivity", "Explicit-Flirting"}

SCORING_MODES = ["raw", "sa", "sa_drop_invalid", "pp_full"]

# prompt_version label (used throughout the CLIs/CSVs here) -> the prompt
# FILE that version actually ran with. This is the only place that mapping
# lives — schema_fields_for_prompt_version below resolves everything else
# through prompt_loader from this file name.
PROMPT_VERSION_TO_FILE = {
    "v1": "prompt.py",
    "v2": "prompt_v2.py",
    "v3": "prompt_v3.py",
    "v4": "prompt_v4.py",
    "v5": "prompt_v5.py",
    "v6": "prompt_v6.py",
    "v7": "prompt_v7.py",
    "v8": "prompt_v8.py",
}

_FIELD_CACHE = {}


def schema_fields_for_prompt_version(prompt_version: str) -> Set[str]:
    """The set of JSON field aliases (t, seg, tr, f, spk, ctx, speech_act,
    j, violation, c, quote_type — whichever this version's schema actually
    declares) via prompt_loader.schema_module_for_prompt. Each schema
    module's raw_model_output_json_schema() has exactly one Flag-shaped
    $defs entry; its properties ARE the field list."""
    if prompt_version in _FIELD_CACHE:
        return _FIELD_CACHE[prompt_version]
    path = PROMPT_VERSION_TO_FILE.get(prompt_version)
    if path is None:
        raise ValueError(f"Unknown prompt_version {prompt_version!r} — add it to PROMPT_VERSION_TO_FILE")
    module = prompt_loader.schema_module_for_prompt(Path(path))
    full_schema = module.raw_model_output_json_schema()
    defs = full_schema.get("$defs", {})
    if not defs:
        raise RuntimeError(f"{path}'s schema module has no $defs — can't determine field availability")
    props = next(iter(defs.values())).get("properties", {})
    fields = set(props.keys())
    _FIELD_CACHE[prompt_version] = fields
    return fields


def has_field(prompt_version: str, field_alias: str) -> bool:
    return field_alias in schema_fields_for_prompt_version(prompt_version)


def adjusted_confidence(flag: dict, mode: str, prompt_version: str) -> Optional[float]:
    """Returns a float, or None meaning "drop this flag" — except in "raw"
    mode, where None means the ordinary thing it's always meant: missing
    confidence (never a drop — "raw" has no drop rule at all). This is
    unambiguous because "sa" and the drop modes always substitute missing
    confidence with 0.0 before anything else, so the ONLY way they can
    still return None is the explicit category-drop branch.

    Rules, applied in order:
      raw:             model_confidence as-is (no substitution, no caps).
      sa:              missing confidence -> 0.0; cap to <=0.3 if
                        speech_act is reported/denial/MISSING (only when
                        the schema has speech_act at all) OR speaker is
                        "unclear" (only when the schema has spk at all).
      sa_drop_invalid: same as sa, then drop if category isn't one of the
                        three real categories.
      pp_full:         same as sa_drop_invalid, then also cap to <=0.3 if
                        violation=="no" (only when the schema has
                        violation at all)."""
    if mode not in SCORING_MODES:
        raise ValueError(f"Unknown scoring mode {mode!r}")

    if mode == "raw":
        return flag.get("model_confidence")

    conf = flag.get("model_confidence")
    if conf is None:
        conf = 0.0

    if has_field(prompt_version, "speech_act"):
        if flag.get("model_speech_act") in ("reported", "denial", None):
            conf = min(conf, 0.3)
    if has_field(prompt_version, "spk"):
        if flag.get("model_speaker") == "unclear":
            conf = min(conf, 0.3)

    if mode == "sa":
        return conf

    if flag.get("category") not in VALID_CATEGORIES:
        return None  # dropped — sa_drop_invalid and pp_full both apply this

    if mode == "sa_drop_invalid":
        return conf

    # pp_full
    if has_field(prompt_version, "violation"):
        if flag.get("model_violation") == "no":
            conf = min(conf, 0.3)
    return conf


def score_flag_all_modes(flag: dict, prompt_version: str, modes=SCORING_MODES) -> dict:
    """Returns the {"raw": ..., "sa": ..., ...} dict stored under a flag's
    "scored" key — every requested mode's adjusted_confidence in one call."""
    return {mode: adjusted_confidence(flag, mode, prompt_version) for mode in modes}
