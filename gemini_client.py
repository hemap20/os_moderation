"""Thin, retrying wrapper around the Gemini API used by Stage 1 and Stage 2."""
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import errors, types

import config

# This pipeline's entire job is to surface harassment/hate/sexual/dangerous
# content for moderation review — Gemini's default safety auto-blocking
# would silently withhold exactly the content we need to see, so it's
# disabled for all categories. (Separate from RECITATION handling above:
# that's a copyright/verbatim-reproduction check, not a harm-category one,
# and BLOCK_NONE here does not affect it.)
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


class GeminiError(Exception):
    pass


class NonRetryableGeminiError(GeminiError):
    """Raised for failures that are deterministic given the same input —
    retrying with backoff would just fail the same way every time (e.g. a
    safety/recitation block on this specific audio's content). Distinct from
    plain GeminiError so call_with_retries can fail fast instead of burning
    through all retries and their backoff delays."""


_NON_RETRYABLE_FINISH_REASONS = {"RECITATION", "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


def get_client() -> genai.Client:
    """Vertex AI auth takes priority when configured (GOOGLE_CLOUD_PROJECT +
    GOOGLE_APPLICATION_CREDENTIALS pointing at a service-account JSON file —
    google-auth picks that file up automatically, we never read/parse it
    ourselves), else falls back to the plain Gemini API key."""
    project = os.environ.get(config.VERTEXAI_PROJECT_ENV)
    if project:
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            raise GeminiError(
                f"{config.VERTEXAI_PROJECT_ENV} is set but GOOGLE_APPLICATION_CREDENTIALS is not — "
                f"point it at your service-account JSON file."
            )
        if not Path(creds_path).is_file():
            raise GeminiError(f"GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {creds_path}")
        location = os.environ.get(config.VERTEXAI_LOCATION_ENV, config.VERTEXAI_DEFAULT_LOCATION)
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get(config.GEMINI_API_KEY_ENV)
    if not api_key:
        raise GeminiError(
            f"Neither {config.VERTEXAI_PROJECT_ENV} (Vertex AI) nor {config.GEMINI_API_KEY_ENV} "
            f"(API key) is set. Put one in your .env / environment."
        )
    return genai.Client(api_key=api_key)


def call_with_retries(
    fn,
    max_retries: int = config.DEFAULT_MAX_RETRIES,
    backoff_base: float = config.DEFAULT_BACKOFF_BASE_SEC,
    on_retry=None,
):
    """Call fn() with exponential backoff. Re-raises the last exception after
    max_retries is exhausted. Never called for the whole run — one file's
    exhaustion just means that file is marked failed by the caller."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except NonRetryableGeminiError:
            raise  # deterministic given this input — retrying wastes the full backoff schedule for nothing
        except errors.ClientError as exc:
            # 400 (bad request — e.g. audio the API can't process) is
            # deterministic for this input and won't succeed on retry. 429
            # (rate limit) and other 4xx that aren't a fixed property of the
            # request itself are still worth retrying with backoff below.
            if getattr(exc, "code", None) == 400:
                raise NonRetryableGeminiError(f"400 INVALID_ARGUMENT (not retryable): {exc}") from exc
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 - deliberately broad, external API
            last_exc = exc

        if attempt == max_retries:
            break
        delay = backoff_base ** attempt
        if on_retry:
            on_retry(attempt, max_retries, delay, last_exc)
        time.sleep(delay)
    raise last_exc


_BOX_2D_ARTIFACT_RE = re.compile(r'"box_2d"\s*:\s*\[')


def repair_transcription_artifacts(text: str) -> str:
    """gemini-3.1-flash-lite occasionally hallucinates a `"box_2d": [`
    key (an object-detection/bounding-box artifact bleeding in from
    unrelated training) in place of `"t": "` for a transcript segment's
    timestamp field, e.g.:
        {"box_2d": [02:21", "text": "..."}   ->   {"t": "02:21", "text": "..."}
    This substitution is mechanically safe (same shape, same trailing
    content) and observed consistently, so it's fixed here rather than
    treated as a parse failure. Any OTHER malformation is left alone —
    guessing at missing brackets/commas elsewhere risks silently
    fabricating data, so those still fail loudly as a per-file error."""
    return _BOX_2D_ARTIFACT_RE.sub('"t": "', text)


def parse_json_lenient(text: str) -> dict:
    """Parse a model response that is supposed to be a single JSON object.
    Tries strict json.loads first; if that fails with trailing/"Extra data"
    (e.g. the model appended stray text or a duplicate object after a valid
    one), falls back to decoding just the first valid JSON object and
    discarding whatever follows, rather than failing the whole file."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        stripped = text.strip()
        try:
            obj, _end = json.JSONDecoder().raw_decode(stripped)
            return obj
        except json.JSONDecodeError:
            raise exc


_MIME_TYPES = {".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/mp4"}


def upload_audio(client: genai.Client, path: Path):
    """Inline bytes rather than the Files API — client.files.upload() only
    works against the Gemini Developer API, not Vertex AI, and inline data
    works identically on both, so this one code path covers either auth
    mode without a runtime branch. Fine at this dataset's file sizes (a few
    MB each, well under the ~20MB inline request limit)."""
    mime_type = _MIME_TYPES.get(path.suffix.lower(), "audio/mp3")
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def generate_text(
    client: genai.Client,
    model: str,
    contents: list,
    response_mime_type: str = "application/json",
    response_json_schema: Optional[dict] = None,
) -> str:
    # gemini-2.5-pro's "thinking" is uncapped by default and can consume the
    # entire output token budget on internal reasoning alone, hitting
    # MAX_TOKENS before writing any actual output (finish_reason=MAX_TOKENS,
    # empty response.text). That failure is deterministic, not transient —
    # capping thinking_budget and giving max_output_tokens headroom avoids
    # burning through all retries on every occurrence.
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type=response_mime_type,
            response_json_schema=response_json_schema,
            temperature=0,
            max_output_tokens=63000,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            safety_settings=SAFETY_SETTINGS,
        ),
    )
    if not response.text:
        # A prompt-level block (response.candidates is None entirely, the
        # reason lives on prompt_feedback) is just as deterministic as a
        # per-candidate finish_reason block — same content, same result
        # every time, so it should fail fast too instead of burning retries.
        block_reason = None
        if response.prompt_feedback is not None:
            block_reason = response.prompt_feedback.block_reason
        block_reason_name = getattr(block_reason, "name", str(block_reason))
        if block_reason_name in _NON_RETRYABLE_FINISH_REASONS:
            raise NonRetryableGeminiError(
                f"Empty response from Gemini (prompt blocked, block_reason={block_reason}): {response}"
            )

        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
        finish_reason_name = getattr(finish_reason, "name", str(finish_reason))
        message = f"Empty response from Gemini (finish_reason={finish_reason}): {response}"
        if finish_reason_name in _NON_RETRYABLE_FINISH_REASONS:
            raise NonRetryableGeminiError(message)
        raise GeminiError(message)
    return response.text
