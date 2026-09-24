"""Loads Stage 2 / Stage 3 prompt text from external, separately-editable files.

Convention observed in the Gemma prompt you supplied (prompt.py at repo root):
a plain Python module exposing ``get_prompt() -> str`` whose returned string
contains a literal ``{json_schema_str}`` placeholder. Note the string is NOT
meant to be passed through ``str.format()`` — it contains other unrelated
curly braces (e.g. the literal example output ``{"d": []}``), which would
break ``.format()``. Instead we do a targeted ``str.replace()`` for each
known placeholder.

To keep Stage 2's (not-yet-written) prompt file compatible with the same
convention, ``prompts/stage2_classification_prompt.py`` follows the same
``get_prompt() -> str`` shape. A plain ``.txt`` file is also supported as a
fallback (no policy logic is hardcoded in this module either way — the
policy/classification text always comes from the external file).
"""
import importlib.util
from pathlib import Path
from typing import Optional

import schemas
import schemas_v4

PLACEHOLDER_JSON_SCHEMA = "{json_schema_str}"
PLACEHOLDER_TRANSCRIPT = "{transcript}"

# Prompt file name -> output schema module. New prompt versions that need
# new output fields get added here, not hardcoded ad hoc in each caller
# (gemma_local.py, gemini_model_eval.py) — this is the ONE place a prompt
# experiment's schema is chosen, so it can never silently drift between the
# two runners. Every prompt NOT listed here (including prompt.py and every
# earlier version) uses the original `schemas` module — ground truth
# (stage2_classify.py / stage12_v2.py) always uses config.STAGE2_PROMPT_PATH
# (prompt.py), which is never in this map, so it always resolves to
# `schemas` — see those scripts' own assertions for the enforced guarantee.
_SCHEMA_MODULE_BY_PROMPT_NAME = {
    "prompt_v4.py": schemas_v4,
}


def schema_module_for_prompt(prompt_path):
    """Returns the schema module (schemas or schemas_v4) a given prompt
    file's output should be generated/parsed against. Looked up by
    filename, not full path, so it works regardless of where the prompt
    file lives (repo root, a synced RunPod copy, etc.)."""
    return _SCHEMA_MODULE_BY_PROMPT_NAME.get(Path(prompt_path).name, schemas)


def _load_py_prompt(path: Path) -> str:
    spec = importlib.util.spec_from_file_location(f"_prompt_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "get_prompt"):
        raise AttributeError(f"{path} does not define get_prompt() -> str")
    return module.get_prompt()


def load_raw_prompt(path: Path) -> str:
    """Load the raw prompt template text, no substitution."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"This pipeline never hardcodes policy/classification text — "
            f"supply this file before running this stage."
        )
    if path.suffix == ".py":
        return _load_py_prompt(path)
    return path.read_text(encoding="utf-8")


def render_prompt(
    path: Path,
    json_schema_str: Optional[str] = None,
    transcript: Optional[str] = None,
) -> str:
    """Load a prompt file and substitute known placeholders via literal
    string replacement (never .format(), to avoid clashing with unrelated
    curly braces the prompt author may use in examples)."""
    text = load_raw_prompt(path)
    if json_schema_str is not None and PLACEHOLDER_JSON_SCHEMA in text:
        text = text.replace(PLACEHOLDER_JSON_SCHEMA, json_schema_str)
    if transcript is not None and PLACEHOLDER_TRANSCRIPT in text:
        text = text.replace(PLACEHOLDER_TRANSCRIPT, transcript)
    return text
