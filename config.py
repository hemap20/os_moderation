"""Central configuration for the moderation eval pipeline."""
import os
from pathlib import Path
from typing import Optional

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

# Classifier model for classify_fps.py's FP-type sub-classification. Would
# ideally be stronger than GEMINI_MODEL (the ground-truth generator) so it
# isn't just repeating ground truth's own blind spots, but gemini-3.5-flash
# was tried and hard-blocks with PROHIBITED_CONTENT on some moderation
# transcripts in this dataset (not overridable via safety_settings) — so
# this reverts to the same model as ground truth per explicit user decision,
# until a stronger tier is confirmed not to hit that block on this content.
GEMINI_CLASSIFIER_MODEL = os.environ.get("GEMINI_CLASSIFIER_MODEL", "gemini-3.1-flash-lite")

# Matching model for analyze_results.py's GT<->model flag pairing step. Same
# reasoning/reversion as GEMINI_CLASSIFIER_MODEL above.
GEMINI_MATCH_MODEL = os.environ.get("GEMINI_MATCH_MODEL", "gemini-3.1-flash-lite")

# --- Vertex AI auth (alternative to the plain API key above). Point
# GOOGLE_APPLICATION_CREDENTIALS at a service-account JSON *file* (never
# inline JSON in .env — see incident notes), and set these two:
VERTEXAI_PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
VERTEXAI_LOCATION_ENV = "GOOGLE_CLOUD_LOCATION"
VERTEXAI_DEFAULT_LOCATION = "global"

# Transcript completeness validation
INCOMPLETE_TRANSCRIPT_MIN_GAP_SEC = 15.0
INCOMPLETE_TRANSCRIPT_GAP_FRACTION = 0.10

DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_SEC = 2.0
DEFAULT_DRY_RUN_LIMIT = 3


def dataset_paths(dataset_root: Optional[str] = None, results_tag: Optional[str] = None) -> dict:
    """Resolve the dataset directory and ITS OWN result directories for
    either the full dataset (default, dataset_root=None -> DOSTT_DIR) or an
    alternate root such as Dostt_dev/ (passed via each script's
    --dataset-root). An alternate root gets its own gemma_results/
    gemini_results/analysis_results directories, suffixed from the root's
    own name (Dostt_dev -> *_dev), so a dev-set run can never read from or
    write into the full-dataset result directories without any script
    needing per-root logic of its own.

    results_tag (--results-tag) adds a further suffix on top, for keeping
    separate PROMPT EXPERIMENTS apart from each other and from the baseline
    — e.g. dataset_root=None, results_tag="promptA" ->
    gemma_results_promptA/, analysis_results_promptA/ (full dataset); or
    dataset_root="Dostt_dev", results_tag="promptA" ->
    gemma_results_dev_promptA/ (dev set). Orthogonal to dataset_root: it
    never affects which dataset_dir is read, only where results are written."""
    if dataset_root is None:
        root = DOSTT_DIR
        suffix = ""
    else:
        root = Path(dataset_root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        name = root.name
        if name == "Dostt":
            suffix = ""
        elif name.startswith("Dostt_"):
            suffix = "_" + name[len("Dostt_"):]
        else:
            suffix = "_" + name
    if results_tag:
        suffix = f"{suffix}_{results_tag}"
    return {
        "dataset_dir": root,
        "gemma_results_dir": PROJECT_ROOT / f"gemma_results{suffix}",
        "gemini_results_dir": PROJECT_ROOT / f"gemini_results{suffix}",
        "analysis_results_dir": PROJECT_ROOT / f"analysis_results{suffix}",
    }
