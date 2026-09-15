"""Central configuration for the moderation eval pipeline."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "Dostt_lite"

# --- New dataset (Dostt) — TP/FP pre-sorted, FN pooled per-language (no
# category yet), TN determined during the pipeline run, never pre-sorted.
DOSTT_DIR = PROJECT_ROOT / "Dostt"
MANIFEST_DIR = PROJECT_ROOT / "manifest"
FN_MANIFEST_PATH = MANIFEST_DIR / "fn_reassignment_manifest.csv"
REVIEW_DIR = PROJECT_ROOT / "review_needed"
TP_MISMATCH_CSV = REVIEW_DIR / "tp_mismatches.csv"
FP_MISMATCH_CSV = REVIEW_DIR / "fp_mismatches.csv"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
TRANSCRIPTS_DIR = GROUND_TRUTH_DIR / "transcripts"
RAW_RESPONSES_DIR = GROUND_TRUTH_DIR / "raw_responses"
CLASSIFICATIONS_DIR = GROUND_TRUTH_DIR / "classifications"
LOGS_DIR = PROJECT_ROOT / "logs"

PROMPTS_DIR = PROJECT_ROOT / "prompts"
# prompt.py is ONE shared classification prompt reused for both stages:
# Stage 2 (Gemini, fed the Stage 1 transcript as text) and Stage 3 (Gemma,
# fed the raw audio). Same policy text, same output schema, different input
# modality — confirmed directly by the user.
CLASSIFICATION_PROMPT_PATH = PROJECT_ROOT / "prompt.py"
STAGE2_PROMPT_PATH = CLASSIFICATION_PROMPT_PATH
STAGE3_PROMPT_PATH = CLASSIFICATION_PROMPT_PATH

# Languages and categories confirmed against the actual Dostt_lite folder names.
LANGUAGES = ["hindi", "tamil", "telugu", "kannada", "malayalam"]
CATEGORIES = ["pm", "sa", "ef"]  # PlatformMove, SuspiciousActivity, Explicit-Flirting
CATEGORY_LABELS = {
    "pm": "PlatformMove",
    "sa": "SuspiciousActivity",
    "ef": "Explicit-Flirting",
}
OUTCOMES = ["TP", "FP", "FN"]  # per (language, category) cell
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
IGNORED_FILENAMES = {".DS_Store"}

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Transcript completeness validation
INCOMPLETE_TRANSCRIPT_MIN_GAP_SEC = 15.0
INCOMPLETE_TRANSCRIPT_GAP_FRACTION = 0.10

DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_SEC = 2.0
DEFAULT_DRY_RUN_LIMIT = 3
