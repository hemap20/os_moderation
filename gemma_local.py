"""Stage 3 (local) — Gemma 4 E2B/E4B native-audio classification.

UNVERIFIED: written against Gemma 4 / transformers documentation only — this
environment has no GPU and cannot download/run the actual model weights, so
none of this has been executed. Expect to debug real runtime issues
yourself; report exact tracebacks back for fixes.

Architecture (per your decisions):
  - Same shared prompt.py as Gemini (Stage 2/3), fed raw audio instead of
    transcript text.
  - Gemma 4 caps audio input at 30s/call, so each file is split into
    <=CHUNK_SECONDS windows and classified INDEPENDENTLY per chunk (native
    audio-in every time, no cross-chunk text context) — a violation split
    exactly across a chunk boundary can be missed; accepted per your choice.
    Per-chunk timestamps are offset-corrected to file-relative time before
    merging.
  - Output fields use a model_-prefixed schema (not ground_truth_), stored
    under model-specific output directories — see schemas_gemma.py.
  - logprob-derived confidence and token entropy: see schemas_gemma.py
    docstring for the exact token-selection method — flagged for your
    review, not just silently baked in.

Setup (not done for you — needs real downloads/hardware):
    python3 -m venv .venv-gemma        # separate venv: transformers/torch
    .venv-gemma/bin/pip install -r requirements-gemma.txt
    brew install ffmpeg                # audio chunking (pydub needs it)
Python 3.9 (this repo's main .venv) is likely too old for current
transformers — use a newer interpreter for .venv-gemma (3.10+).

Usage:
    python3 gemma_local.py --model e2b --dry-run
    python3 gemma_local.py --model e2b
    python3 gemma_local.py --model e4b --thinking
    python3 gemma_local.py --model e2b --force --limit 5
"""
import argparse
import json
import re
import time
import traceback
from pathlib import Path
from typing import List, Optional

import config
import dataset_v2 as dsv2
import prompt_loader
from pipeline_logging import StageLogger
from schemas_gemma import GemmaChunkFlag, GemmaFileResult

MODEL_IDS = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
    "12b": "google/gemma-4-12B-it",
}

CHUNK_SECONDS_MAX = 30.0
DEFAULT_CHUNK_SECONDS = 28.0  # headroom under the 30s hard cap
TOP_K_LOGPROBS = 20

# Reassigned by main() alongside PROMPT_PATH — the schema sent to the model
# must match whichever prompt is active (see prompt_loader.
# schema_module_for_prompt). Default here is schemas.py's (v1) schema, same
# as PROMPT_PATH's default.
RAW_SCHEMA_STR = json.dumps(prompt_loader.schema_module_for_prompt(config.CLASSIFICATION_PROMPT_PATH).raw_model_output_json_schema())


# Mutable module globals (not frozen constants) — reassigned by main() from
# --dataset-root via config.dataset_paths(), so a dev-set run's outputs land
# under gemma_results_<suffix>/ and its dataset reads from the alternate
# root, without any other function in this module needing to change.
DATASET_DIR: Path = config.DOSTT_DIR
GEMMA_RESULTS_DIR: Path = config.PROJECT_ROOT / "gemma_results"
# Reassigned by main() from --prompt-path — lets a prompt experiment run
# against an alternate file (e.g. prompt_v2.py) WITHOUT touching prompt.py
# itself, since prompt.py is also what generated ground truth and must stay
# fixed for scoring to remain meaningful.
PROMPT_PATH: Path = config.CLASSIFICATION_PROMPT_PATH


def output_dir(model_key: str, thinking: bool) -> Path:
    suffix = "thinking" if thinking else "nothinking"
    return GEMMA_RESULTS_DIR / f"{model_key}_{suffix}"


# ---------------------------------------------------------------------------
# Model loading (lazy — only imports torch/transformers when actually run,
# so --help and unrelated commands don't require the heavy deps installed)
# ---------------------------------------------------------------------------
def load_model(model_key: str):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_id = MODEL_IDS[model_key]
    processor = AutoProcessor.from_pretrained(model_id)
    device_map = "mps" if torch.backends.mps.is_available() else "auto"
    # SDPA dispatches to a fused/flash-equivalent CUDA kernel automatically
    # (PyTorch 2.0+, no extra package) — pure speedup, mathematically
    # equivalent output to eager attention, so it can't affect any of the
    # logprob/entropy/confidence extraction downstream. Not available on
    # MPS (Apple's backend doesn't implement the fused kernel), so only
    # requested on CUDA — "auto" (the default) picks eager there, which is
    # the correct, working fallback we've already validated on Mac.
    attn_kwargs = {} if torch.backends.mps.is_available() else {"attn_implementation": "sdpa"}
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype="auto", device_map=device_map, **attn_kwargs)
    if attn_kwargs:
        actual = getattr(model.config, "_attn_implementation", "unknown")
        print(f"[load_model] requested attn_implementation={attn_kwargs['attn_implementation']!r}, model reports {actual!r}", flush=True)
    return model, processor


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------
def chunk_audio(path: Path, chunk_seconds: float, out_dir: Path) -> List[tuple]:
    """Splits audio into <=chunk_seconds WAV chunks (Gemma's docs use WAV
    examples; chunking also sidesteps needing the model to handle arbitrary
    mp3 internals). Returns [(chunk_path, start_offset_sec), ...].
    Requires ffmpeg (via pydub) — see module docstring for setup."""
    from pydub import AudioSegment

    if chunk_seconds > CHUNK_SECONDS_MAX:
        raise ValueError(f"chunk_seconds={chunk_seconds} exceeds Gemma 4's {CHUNK_SECONDS_MAX}s cap")

    audio = AudioSegment.set_channels(AudioSegment.from_file(path), 1).set_frame_rate(16000)
    total_ms = len(audio)
    chunk_ms = int(chunk_seconds * 1000)

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for i, start_ms in enumerate(range(0, total_ms, chunk_ms)):
        end_ms = min(start_ms + chunk_ms, total_ms)
        segment = audio[start_ms:end_ms]
        chunk_path = out_dir / f"{path.stem}_chunk{i:03d}.wav"
        segment.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_ms / 1000.0))
    return chunks


# ---------------------------------------------------------------------------
# Logprobs / entropy
# ---------------------------------------------------------------------------
def token_logprobs_and_entropy(model, processor, input_len: int, generated_ids, scores):
    """For each generated step: exact full-vocabulary log-softmax (not a
    top-k approximation), plus the top-K alternative tokens for inspection.
    Returns a list of dicts: {token, token_id, logprob, entropy, topk:[{token,logprob}]}."""
    import torch

    out = []
    for step, logits in enumerate(scores):
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        chosen_id = generated_ids[input_len + step].item()
        chosen_logprob = log_probs[chosen_id].item()
        # Exact entropy: -sum(p * log p) over the full vocabulary distribution.
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()
        topk_logprobs, topk_ids = torch.topk(log_probs, TOP_K_LOGPROBS)
        topk = [
            {"token": processor.decode([tid]), "logprob": lp.item()}
            for tid, lp in zip(topk_ids.tolist(), topk_logprobs)
        ]
        out.append({
            "token": processor.decode([chosen_id]),
            "token_id": chosen_id,
            "logprob": chosen_logprob,
            "entropy": entropy,
            "topk": topk,
        })
    return out


def find_token_span_for_substring(token_infos: list, full_text: str, substring: str) -> Optional[tuple]:
    """Locate which generated-token indices correspond to a character span
    in full_text, by progressively concatenating decoded tokens and matching
    character offsets. Approximate but robust to subword tokenization —
    returns (start_idx, end_idx) inclusive, or None if not found."""
    if not substring:
        return None
    char_idx = full_text.find(substring)
    if char_idx == -1:
        return None
    end_char_idx = char_idx + len(substring)

    cursor = 0
    start_tok, end_tok = None, None
    for i, info in enumerate(token_infos):
        tok_text = info["token"]
        tok_start, tok_end = cursor, cursor + len(tok_text)
        if start_tok is None and tok_end > char_idx:
            start_tok = i
        if tok_start < end_char_idx:
            end_tok = i
        cursor = tok_end
        if tok_start >= end_char_idx:
            break
    if start_tok is None or end_tok is None:
        return None
    return (start_tok, end_tok)


def compute_flag_confidence_and_entropy(token_infos: list, full_text: str, flag_start_char: int, category_value: str, excerpt_value: str):
    """decision token = first generated token of this flag object (the
    token at/just after flag_start_char, i.e. right after the preceding '['
    or ',' — where the model commits to a new entry). category token span =
    the tokens covering the 'f' field's value. Confidence = geometric mean
    of exp(logprob) of the decision token and the MEAN exp(logprob) across
    the category token span."""
    import math as _math

    cursor = 0
    decision_idx = None
    for i, info in enumerate(token_infos):
        tok_text = info["token"]
        if decision_idx is None and cursor + len(tok_text) > flag_start_char:
            decision_idx = i
            break
        cursor += len(tok_text)
    decision_logprob = token_infos[decision_idx]["logprob"] if decision_idx is not None else None

    cat_span = find_token_span_for_substring(token_infos, full_text, category_value)
    category_logprob = None
    if cat_span:
        s, e = cat_span
        lps = [token_infos[i]["logprob"] for i in range(s, e + 1)]
        category_logprob = sum(lps) / len(lps) if lps else None

    confidence = None
    if decision_logprob is not None and category_logprob is not None:
        confidence = _math.sqrt(_math.exp(decision_logprob) * _math.exp(category_logprob))

    excerpt_entropy = None
    exc_span = find_token_span_for_substring(token_infos, full_text, excerpt_value)
    if exc_span:
        s, e = exc_span
        ent_values = [token_infos[i]["entropy"] for i in range(s, e + 1)]
        excerpt_entropy = {
            "per_token": ent_values,
            "mean": sum(ent_values) / len(ent_values) if ent_values else None,
        }

    return {
        "decision_logprob": decision_logprob,
        "category_logprob": category_logprob,
        "logprob_derived_confidence": confidence,
        "excerpt_token_entropy": excerpt_entropy,
    }


def find_violation_token_index(token_infos: list, full_text: str, flag_start_char: int, flag_end_char: int) -> Optional[int]:
    """Locate the token covering the FIRST character of the "violation"
    field's value, within THIS flag's own JSON object only (bounded by
    flag_start_char/flag_end_char from _find_flag_entry_char_spans, so a
    "violation" key belonging to a later flag can never be matched here) —
    same span-bounding principle as the category lookup in
    compute_flag_confidence_and_entropy, just anchored on a literal key
    name instead of a value substring, since the value itself ("yes"/"no")
    is exactly what we're trying to locate the token for."""
    segment = full_text[flag_start_char:flag_end_char + 1]
    key_idx = segment.find('"violation"')
    if key_idx == -1:
        return None
    colon_idx = segment.find(":", key_idx)
    if colon_idx == -1:
        return None
    quote_idx = segment.find('"', colon_idx)
    if quote_idx == -1:
        return None
    value_start_abs = flag_start_char + quote_idx + 1  # first char of yes/no, right after the opening quote

    cursor = 0
    for i, info in enumerate(token_infos):
        tok_end = cursor + len(info["token"])
        if tok_end > value_start_abs:
            return i
        cursor = tok_end
    return None


def compute_violation_probability(token_infos: list, idx: Optional[int]) -> dict:
    """Given the violation-token index (or None if not found), returns
    {logprob_violation, p_violation_yes, p_violation_method,
    violation_token_topk}. Three ways p_violation_yes can be derived (see
    schemas_gemma.GemmaChunkFlag.p_violation_method for the exact
    semantics of each) — token text is matched leniently (stripped of
    surrounding quote/space characters, case-insensitive prefix check)
    since exact subword tokenization of `"yes`/`"no` at a JSON quote
    boundary isn't guaranteed to isolate the bare word cleanly."""
    import math

    if idx is None or idx >= len(token_infos):
        return {"logprob_violation": None, "p_violation_yes": None,
                "p_violation_method": None, "violation_token_topk": None}

    info = token_infos[idx]
    logprob_violation = info["logprob"]
    topk = info.get("topk", [])
    token_text = info["token"].strip(" \"'").lower()

    if token_text.startswith("yes"):
        p_yes = math.exp(logprob_violation)
        method = "direct"
    elif token_text.startswith("no"):
        yes_entry = next(
            (t for t in topk if t["token"].strip(" \"'").lower().startswith("yes")), None
        )
        if yes_entry is not None:
            p_yes = math.exp(yes_entry["logprob"])
            method = "topk_yes"
        else:
            p_yes = 1.0 - math.exp(logprob_violation)
            method = "complement"
    else:
        # Unexpected token text at the located position — don't guess.
        p_yes = None
        method = "unknown_token"

    return {
        "logprob_violation": logprob_violation,
        "p_violation_yes": p_yes,
        "p_violation_method": method,
        "violation_token_topk": topk,
    }


# ---------------------------------------------------------------------------
# Per-chunk classification call
# ---------------------------------------------------------------------------
def classify_chunk(model, processor, chunk_path: Path, thinking: bool, logger: StageLogger):
    import torch

    prompt_text = prompt_loader.render_prompt(PROMPT_PATH, json_schema_str=RAW_SCHEMA_STR)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "audio", "audio": str(chunk_path)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=thinking,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=4096, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    raw_response = processor.decode(generated_ids[input_len:], skip_special_tokens=False)

    if thinking:
        # parse_response needs the exact prompt text that preceded
        # generation (the chat template may pre-write an opening <think>
        # tag as part of the assistant turn) — reconstruct it by decoding
        # the actual input tokens, not by re-rendering the template, so it
        # exactly matches what the model saw.
        prefix_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        parsed = processor.parse_response(raw_response, prefix=prefix_text)
        answer_text = parsed.get("content", raw_response)
        thinking_text = parsed.get("thinking")
    else:
        answer_text = processor.decode(generated_ids[input_len:], skip_special_tokens=True)
        thinking_text = None

    token_infos = token_logprobs_and_entropy(model, processor, input_len, generated_ids, outputs.scores)

    return answer_text, thinking_text, raw_response, token_infos


def _strip_code_fence(text: str) -> str:
    """Gemini's response_mime_type=application/json enforces raw JSON with
    no wrapper — local transformers generation has no equivalent constraint,
    and Gemma reliably wraps output in a ```json ... ``` markdown fence.
    Strip that before attempting to parse."""
    import re

    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", stripped, re.DOTALL)
    return match.group(1) if match else stripped


def parse_json_lenient(text: str) -> dict:
    """Same lenient-parse approach as gemini_client.parse_json_lenient, but
    duplicated here (not imported) — gemini_client.py imports google.genai
    at module level, which only exists in the main pipeline's .venv, not
    .venv-gemma. These two pipelines' dependencies must stay independent."""
    text = _strip_code_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            obj, _end = json.JSONDecoder().raw_decode(text.strip())
            return obj
        except json.JSONDecodeError:
            raise exc


# Fields whose absence is tracked per-file (see GemmaFileResult.
# missing_field_counts) — "c" pre-dates v4 (already ~10% missing on
# e2b_nothinking across every prompt version); the other three only ever
# appear under prompt_v4.py's schema, but tracking them unconditionally is
# harmless (they're just never present, hence never "missing" in a
# meaningful sense, for older prompts — see GemmaFileResult's docstring on
# a field being counted with 0 vs not present in the dict at all).
_MISSING_TRACKED_FIELDS = ["c", "speech_act", "quote_type", "violation"]


def parse_and_score_flags(answer_text: str, token_infos: list, chunk_offset_sec: float, logger: StageLogger) -> tuple:
    """Returns (flags, missing_field_counts) — see _MISSING_TRACKED_FIELDS."""
    missing_counts = {k: 0 for k in _MISSING_TRACKED_FIELDS}
    try:
        parsed = parse_json_lenient(answer_text)
    except Exception as exc:
        logger.warn(f"chunk classification JSON parse failed: {exc}")
        return [], missing_counts

    flags = []
    # The model occasionally returns a bare JSON array (the "d" list itself)
    # instead of the expected {"d": [...]} object — recover rather than crash.
    if isinstance(parsed, list):
        logger.warn("model returned a bare JSON array instead of {\"d\": [...]}; treating it as the flags list")
        raw_flags = parsed
    elif isinstance(parsed, dict):
        raw_flags = parsed.get("d", [])
    else:
        logger.warn(f"model returned unexpected JSON top-level type {type(parsed).__name__}, treating as no flags")
        raw_flags = []
    entry_spans = _find_flag_entry_char_spans(answer_text, len(raw_flags))
    for flag, (start_char, end_char) in zip(raw_flags, entry_spans):
        scoring = compute_flag_confidence_and_entropy(
            token_infos, answer_text, start_char, flag.get("f", ""), flag.get("seg", "")
        )

        for key in _MISSING_TRACKED_FIELDS:
            if flag.get(key) is None:
                missing_counts[key] += 1

        violation_idx = find_violation_token_index(token_infos, answer_text, start_char, end_char)
        violation_scoring = compute_violation_probability(token_infos, violation_idx)

        try:
            t_sec = _parse_mmss(flag.get("t", ""))
        except Exception:
            t_sec = None
        if t_sec is not None:
            global_ts = _format_mmss(t_sec + chunk_offset_sec)
        else:
            logger.warn(
                f"chunk@{chunk_offset_sec}s: model returned non-MM:SS timestamp "
                f"{flag.get('t')!r} for category {flag.get('f')!r} — using it as-is, "
                f"NOT offset-corrected to file-relative time."
            )
            global_ts = flag.get("t", "")
        flags.append(GemmaChunkFlag(
            model_category=flag.get("f") or "",
            model_timestamp=global_ts,
            model_excerpt=flag.get("seg") or "",
            model_translation=flag.get("tr") or "",
            model_justification=flag.get("j") or "",
            model_confidence=flag.get("c"),
            model_speech_act=flag.get("speech_act"),
            model_quote_type=flag.get("quote_type"),
            model_violation=flag.get("violation"),
            logprob_violation=violation_scoring["logprob_violation"],
            p_violation_yes=violation_scoring["p_violation_yes"],
            p_violation_method=violation_scoring["p_violation_method"],
            violation_token_topk=violation_scoring["violation_token_topk"],
            logprob_decision=scoring["decision_logprob"],
            logprob_category=scoring["category_logprob"],
            logprob_derived_confidence=scoring["logprob_derived_confidence"],
            excerpt_token_entropy=scoring["excerpt_token_entropy"],
        ))
    return flags, missing_counts


def _find_flag_entry_char_spans(text: str, n_flags: int) -> List[tuple]:
    """(start, end) char offsets of each '{'...'}' entry inside the top-level
    "d" array, via a simple bracket-depth scan (avoids re-parsing with a
    position-tracking JSON decoder). end is the index of the matching '}',
    inclusive — used both as the "flag start" anchor for the decision-token
    lookup and to BOUND the violation-token search to this flag's own
    object (so a "violation" key value from a LATER flag can never be
    mistaken for this one's)."""
    d_idx = text.find('"d"')
    if d_idx == -1:
        return [(0, len(text) - 1)] * n_flags
    arr_start = text.find("[", d_idx)
    if arr_start == -1:
        return [(0, len(text) - 1)] * n_flags

    spans = []
    depth = 0
    cur_start = None
    i = arr_start
    in_string = False
    escape = False
    while i < len(text) and len(spans) < n_flags:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    cur_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and cur_start is not None:
                    spans.append((cur_start, i))
                    cur_start = None
            elif ch == "]" and depth == 0:
                break
        i += 1
    while len(spans) < n_flags:
        spans.append(spans[-1] if spans else (arr_start, len(text) - 1))
    return spans


def _find_flag_entry_char_offsets(text: str, n_flags: int) -> List[int]:
    """Back-compat wrapper: just the start offsets from
    _find_flag_entry_char_spans, for callers that only need the decision-
    token anchor and don't need the end bound."""
    return [s for s, _e in _find_flag_entry_char_spans(text, n_flags)]


_BARE_SECONDS_RE = re.compile(r"^\d+(\.\d+)?s$")


def _parse_mmss(ts: str) -> Optional[float]:
    """Accepts MM:SS, H:MM:SS (original), plus two malformed-but-common
    model outputs observed in practice: "Ns" (bare seconds with an 's'
    suffix, e.g. "5s", "84.0s") and bare "N" (a plain number, no colon, no
    suffix, e.g. "0", "11"). All of these previously fell through to
    returning None, which meant the caller skipped the chunk-offset
    correction entirely and kept the model's chunk-relative value as if it
    were already file-relative — silently losing real catches whenever the
    model wrote a non-MM:SS timestamp in a chunk beyond the first."""
    ts = ts.strip()
    if not ts:
        return None
    if _BARE_SECONDS_RE.match(ts):
        return float(ts[:-1])
    if ":" in ts:
        try:
            parts = [float(p) for p in ts.split(":")]
        except ValueError:
            return None
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return None
    try:
        return float(ts)
    except ValueError:
        return None


def _format_mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------
def process_file(model, processor, record, model_key: str, thinking: bool, chunk_seconds: float, logger: StageLogger, tmp_dir: Path) -> GemmaFileResult:
    chunks = chunk_audio(record.path, chunk_seconds, tmp_dir)
    all_flags: List[GemmaChunkFlag] = []
    raw_responses = []
    n_failed = 0
    file_missing_counts = {k: 0 for k in _MISSING_TRACKED_FIELDS}

    for chunk_path, offset_sec in chunks:
        chunk_t0 = time.time()
        try:
            answer_text, thinking_text, raw_response, token_infos = classify_chunk(model, processor, chunk_path, thinking, logger)
            elapsed = time.time() - chunk_t0
            # A single chunk normally takes ~10-25s (non-thinking) or up to
            # ~90s (thinking). A chunk taking minutes usually means the
            # system is swap-thrashing (e.g. another heavy model process
            # running concurrently) rather than anything wrong with this
            # chunk itself — surfacing it loudly and immediately (not just
            # visible in hindsight from log timestamp gaps) is what lets you
            # notice and intervene in real time instead of losing hours.
            if elapsed > 120:
                logger.warn(
                    f"{record.file_id} chunk@{offset_sec}s took {elapsed:.0f}s (normal is ~10-90s) — "
                    f"likely system memory pressure (check for other concurrent model processes), not a bug in this chunk."
                )
            raw_responses.append({"chunk_offset_sec": offset_sec, "raw": raw_response, "thinking": thinking_text})
            flags, chunk_missing_counts = parse_and_score_flags(answer_text, token_infos, offset_sec, logger)
            all_flags.extend(flags)
            for key, n in chunk_missing_counts.items():
                file_missing_counts[key] += n
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            logger.error(f"{record.file_id} chunk@{offset_sec}s failed: {exc}\n{traceback.format_exc()}")
        finally:
            chunk_path.unlink(missing_ok=True)

    # A file where every chunk failed produced zero real signal — reporting
    # that as status="success" with flags=[] would be indistinguishable from
    # a genuine "no violations found", which is a materially different and
    # much more important thing to know when reviewing results later.
    n_total = len(chunks)
    if n_total > 0 and n_failed == n_total:
        status, error = "error", f"all {n_total} chunk(s) failed — see log"
    else:
        status, error = "success", None

    return GemmaFileResult(
        file_id=record.file_id,
        model=model_key,
        thinking=thinking,
        chunk_seconds=chunk_seconds,
        flags=all_flags,
        chunks_total=n_total,
        chunks_failed=n_failed,
        status=status,
        error=error,
        missing_field_counts=file_missing_counts,
    ), raw_responses


def already_done(model_key: str, thinking: bool, file_id: str) -> bool:
    p = output_dir(model_key, thinking) / "results" / f"{file_id}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("status") == "success"
    except Exception:
        return False


def write_result(model_key: str, thinking: bool, result: GemmaFileResult, raw_responses: list):
    base = output_dir(model_key, thinking)
    (base / "results").mkdir(parents=True, exist_ok=True)
    (base / "raw_responses").mkdir(parents=True, exist_ok=True)
    (base / "results" / f"{result.file_id}.json").write_text(result.model_dump_json(indent=2))
    (base / "raw_responses" / f"{result.file_id}.json").write_text(
        json.dumps(raw_responses, indent=2, ensure_ascii=False)
    )


def write_error(model_key: str, thinking: bool, file_id: str, error: str):
    base = output_dir(model_key, thinking)
    (base / "results").mkdir(parents=True, exist_ok=True)
    err = GemmaFileResult(file_id=file_id, model=model_key, thinking=thinking, chunk_seconds=0, flags=[], status="error", error=error)
    (base / "results" / f"{file_id}.json").write_text(err.model_dump_json(indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["e2b", "e4b", "12b"], required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=config.DEFAULT_DRY_RUN_LIMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-root", type=str, default=None,
                         help="Alternate dataset root, e.g. Dostt_dev — routes results to "
                              "gemma_results_<suffix>/ automatically; default uses the full Dostt/ dataset")
    parser.add_argument("--results-tag", type=str, default=None,
                         help="Extra results-directory suffix for a prompt experiment, e.g. "
                              "--results-tag promptA -> gemma_results_promptA/")
    parser.add_argument("--prompt-path", type=str, default=None,
                         help="Alternate prompt file for a prompt experiment, e.g. prompt_v2.py — "
                              "does NOT affect ground truth, which always used prompt.py; "
                              "default (unset) uses config.CLASSIFICATION_PROMPT_PATH (prompt.py)")
    args = parser.parse_args()

    global DATASET_DIR, GEMMA_RESULTS_DIR, PROMPT_PATH, RAW_SCHEMA_STR
    paths = config.dataset_paths(args.dataset_root, args.results_tag)
    DATASET_DIR = paths["dataset_dir"]
    if args.prompt_path:
        p = Path(args.prompt_path)
        PROMPT_PATH = p if p.is_absolute() else config.PROJECT_ROOT / p
    schema_module = prompt_loader.schema_module_for_prompt(PROMPT_PATH)
    RAW_SCHEMA_STR = json.dumps(schema_module.raw_model_output_json_schema())
    GEMMA_RESULTS_DIR = paths["gemma_results_dir"]

    logger = StageLogger(f"gemma_local_{args.model}_{'thinking' if args.thinking else 'nothinking'}")
    logger.info(f"Loading {MODEL_IDS[args.model]} (thinking={args.thinking})...")
    model, processor = load_model(args.model)
    logger.info("Model loaded.")

    records = dsv2.load_dataset_v2(DATASET_DIR)
    if args.dry_run:
        records = records[: args.dry_run_limit]
        logger.info(f"DRY RUN: {len(records)} file(s)")
    else:
        if not args.force:
            before = len(records)
            records = [r for r in records if not already_done(args.model, args.thinking, r.file_id)]
            logger.info(f"Skipping {before - len(records)} already-done file(s)")
        if args.limit:
            records = records[: args.limit]

    # Nested inside this run's OWN output_dir (per model + thinking flag), not
    # the shared GEMMA_RESULTS_DIR — chunk filenames are only unique per
    # file_id, so two runs (e.g. --thinking and non-thinking) sharing one tmp
    # dir race on the same chunk*.wav paths when run concurrently, corrupting
    # whichever one loses the race (observed: "Incorrect padding" audio
    # errors from a chunk truncated mid-write by the other process).
    tmp_dir = output_dir(args.model, args.thinking) / "_chunks_tmp"
    n_success, n_error = 0, 0
    total_missing_counts = {k: 0 for k in _MISSING_TRACKED_FIELDS}
    t0 = time.time()

    for i, record in enumerate(records, 1):
        logger.info(f"[{i}/{len(records)}] {record.file_id}")
        try:
            result, raw_responses = process_file(model, processor, record, args.model, args.thinking, args.chunk_seconds, logger, tmp_dir)
            if result.status == "success":
                n_success += 1
            else:
                n_error += 1
            for key, n in result.missing_field_counts.items():
                total_missing_counts[key] = total_missing_counts.get(key, 0) + n
            if args.dry_run:
                print(result.model_dump_json(indent=2))
            else:
                write_result(args.model, args.thinking, result, raw_responses)
        except Exception as exc:  # noqa: BLE001
            n_error += 1
            logger.error(f"{record.file_id}: FAILED — {exc}\n{traceback.format_exc()}")
            if not args.dry_run:
                write_error(args.model, args.thinking, record.file_id, str(exc))

    missing_summary = ", ".join(f"{k}={v}" for k, v in total_missing_counts.items())
    logger.info(f"Done in {time.time() - t0:.1f}s — success={n_success} error={n_error} — missing fields (count of flags lacking each): {missing_summary}")


if __name__ == "__main__":
    main()
