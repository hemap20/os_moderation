# Gemma vs Gemini moderation eval pipeline

## Setup
```
.venv/bin/pip install -r requirements.txt
# .env already has GEMINI_API_KEY
```

## Dataset
`Dostt_lite/{language}_{category}/{TP,FP,FN}/...` (TP further split into
`Single`/`Multi`), plus `Dostt_lite/TN/*.mp3` pooled across everything.
Categories: `pm` (PlatformMove), `sa` (SuspiciousActivity), `ef` (Explicit-Flirting).
Filenames are opaque — the full filename is the `file_id` everywhere.

Run `.venv/bin/python3 dataset.py` any time to see per-cell file counts and
which (language × category × outcome) cells are currently empty. Right now
that's all 5 language × SA FP cells (hindi/tamil/telugu/kannada/malayalam),
not just one — Stage 4's metrics code will need to report "N/A — no FP data"
for every one of those, generically, not just a single hardcoded cell.

## Stage 1 — transcription (you run this)
```
.venv/bin/python3 stage1_transcribe.py --dry-run     # 2-3 files, prints only, writes nothing
.venv/bin/python3 stage1_transcribe.py                 # full run, resumable, incremental writes
.venv/bin/python3 stage1_transcribe.py --force         # redo everything
.venv/bin/python3 stage1_transcribe.py --limit 20      # cap this run to 20 files
```
Output: `ground_truth/transcripts/{file_id}.json`,
`ground_truth/raw_responses/{file_id}_transcription.json`. Logs to
`logs/stage1_transcribe.log`. Flags `incomplete_transcript: true` whenever the
transcript ends more than `max(15s, 10% of duration)` before the audio does.

## Stage 2 — ground truth classification (you run this)
Needs `prompts/stage2_classification_prompt.py` filled in first (it currently
raises `NotImplementedError` by design — no policy logic is hardcoded
anywhere in this pipeline). Edit that file's `get_prompt()` to return your
classification prompt text; it can use the literal placeholders
`{transcript}` and `{json_schema_str}` (substituted via `str.replace`, not
`.format()`, so any other braces in your prompt — JSON examples, etc. — are
left alone).
```
.venv/bin/python3 stage2_classify.py --dry-run
.venv/bin/python3 stage2_classify.py
```
Output: `ground_truth/classifications/{file_id}.json` with every field
prefixed `ground_truth_`. Only proceed to Stage 3/4 once you've reviewed this
output yourself.

## Stage 3 / Stage 4 (not yet built)
Per the agreed build order, these are built next, as a single unattended
RunPod command with pre-flight validation + smoke test + resumable Gemma
inference + metrics, only after you confirm Stage 1-2 ground truth is
correct and complete.
