# Task: Sub-classify false-positive (FP) flags by error type

## Context

This repo evaluates open-source models (Gemma variants) against gemini-3.5-flash-lite on audio moderation of Indian-language calls. The categories are PlatformMove, SuspiciousActivity and Explicit-Flirting, and the policy prompt is `prompt.py`.

Ground truth comes from a Gemini pipeline:
- `stage1_transcribe.py` produces a transcript per file.
- `stage2_classify.py` produces ground-truth flags per file.

`analyze_results.py` then uses Gemini to match each model's flags to the ground-truth flags. Any model flag with no matching ground-truth flag is counted as an FP (`"matched": false` in `analysis_results/<model>/per_file/*.json`).

Today every FP is counted the same way. I want each FP flag labelled with **why** it is wrong, so I can choose the right prompt fix for each error type. Read `analyze_results.py`, `gemini_client.py`, `config.py`, `dataset_v2.py`, `schemas_gemma.py` and `prompt.py` before writing anything, and reuse their existing loaders, client, retry logic and logging conventions.

## What counts as an FP

Work at the **flag level**. Every model flag with `matched == false` gets classified. That includes extra unmatched flags inside files whose `file_bucket` is TP, not only flags in FP files. Record `file_bucket` alongside each classified flag so I can separate the two cases later.

Important: the `Dostt/<lang>_<cat>/{TP,FP,FN}` folder names are the dataset's original labels. They are **not** the model's outcome. Never use the folder name as the FP label. It is fine to pass it to the classifier as extra context, clearly named `dataset_origin_label`.

## Error-type taxonomy

Assign exactly one `fp_type` per FP flag. Apply the checks **in the order below** and stop at the first one that fits. The order matters because several types can look alike, and the earlier checks rule out causes that would make the later ones meaningless.

1. **`GT_MISS`**: The model is actually right, and the ground truth missed a real violation. The quote is present in the transcript, it clearly meets the policy definition of the stated category, and no exception in `prompt.py` applies.
   - Use this label conservatively. Ground truth is also Gemini, so every `GT_MISS` goes into the human-review export (see below).

2. **`WRONG_CATEGORY`**: The same (or essentially the same) utterance *is* flagged in the ground truth, but under a different category. The content was real and violating; only the category is wrong.

3. **`DUPLICATE_OR_EXTRA_INSTANCE`**: The flag refers to a violation that is already matched to a ground-truth flag in this file. It is either the same utterance repeated, or another instance of the same violation that the ground truth represented with one flag (for example, the phone number spoken twice).
   - This is a scoring artifact, not a model error. Keep it separate so it can be excluded from "real" FPs.

4. **`FABRICATION`**: Neither the quoted content nor anything resembling it appears anywhere in the ground-truth transcript, including near the timestamp. The model invented it.

5. **`MISHEARING`**: Something similar-sounding or partially overlapping *is* in the transcript near the timestamp, but the model's version changes the meaning in a way that creates the violation. Typical cases are homophones, a code-mixed word misread, or a number or app name misheard.
   - The distinction from `FABRICATION` is whether there is a real source utterance the model distorted.

6. **`EXCEPTION_IGNORED`**: The quote is real and accurately understood, but it matches an explicit "Do NOT flag / NEVER FLAG / Override" item in `prompt.py`. Examples: an in-platform video-call request, coins or gifts, a denial or refusal to share contact details, "I can't hear you".
   - Record *which* exception in `fp_exception_matched`, quoted from `prompt.py`.

7. **`CONTEXT_CONFUSION`**: The quote is real and would be a violation in isolation, but the context shows it isn't happening now. Examples: a hypothetical, reported speech about someone else, a joke, a negation, a question that was declined, the expert warning the user, or a description of a general practice. It is also not covered by an explicit exception (otherwise it would be type 6).

8. **`OVER_SENSITIVITY`**: The quote is real, accurately understood and in context, but it doesn't reach the violation threshold. Examples: mild romance or compliments flagged as Explicit-Flirting, general money talk flagged as SuspiciousActivity, sharing a city or name flagged as PlatformMove. No explicit exception names it, but the category definition doesn't cover it.

9. **`UNCLEAR`**: Evidence is insufficient to decide. Examples: the transcript is missing or incomplete around the timestamp (`incomplete_transcript: true`), the timestamp is invalid, or the quote is too garbled.
   - Always state what's missing in the reason.

## How to classify each flag

For each FP flag, build one classification request containing:
- the model flag in full, including category, timestamp, native-language excerpt, translation, justification and self-reported confidence. Load the excerpt from the model's raw result files, because `per_file` currently drops it.
- the ground-truth transcript. Include the full transcript if it fits; otherwise include a window of about ±90 s around the flag timestamp, plus the full transcript text searched for the excerpt's key terms.
- **all** ground-truth flags for that file, with their `matched` status against this model's flags.
- the **other** model flags in the same file (needed for the duplicate check).
- the policy definitions and exception lists, loaded from `prompt.py` at runtime rather than copied.
- `language` and `dataset_origin_label`.

Use Gemini through the existing `gemini_client`. Make the model name a CLI argument (`--classifier-model`), defaulting to the strongest Gemini model already configured in `config.py`. It should be a stronger model than the one that produced the ground truth, so it isn't just repeating the ground truth's blind spots.

Use a JSON response schema. The classifier prompt must tell the model to:
- walk the decision order 1→9 explicitly and stop at the first match;
- write its reasoning in a `reasoning` field **before** giving `fp_type`;
- quote the exact transcript text it relied on in `evidence_transcript_quote`, or an empty string if nothing was found (which is required for `FABRICATION`);
- remember that it is seeing a transcript, not audio. `FABRICATION` and `MISHEARING` are judged against the ground-truth transcript, which may itself contain errors. It should set `transcript_reliability: "ok" | "suspect"` when the transcript itself looks garbled near the timestamp.

Classifier output schema, one object per flag:

```json
{
  "reasoning": "string, step-by-step walk through the decision order",
  "fp_type": "GT_MISS | WRONG_CATEGORY | DUPLICATE_OR_EXTRA_INSTANCE | FABRICATION | MISHEARING | EXCEPTION_IGNORED | CONTEXT_CONFUSION | OVER_SENSITIVITY | UNCLEAR",
  "fp_reason": "one sentence, English",
  "evidence_transcript_quote": "string",
  "evidence_timestamp": "mm:ss or empty",
  "fp_exception_matched": "string or empty",
  "related_gt_flag_index": "int or null",
  "transcript_reliability": "ok | suspect",
  "classifier_confidence": "low | medium | high"
}
```

Batch requests the same way `analyze_results.py` batches matching, and keep the batch size configurable. Validate every response against the enum, and retry or mark `UNCLEAR` on invalid output.

## Where to store results

Do **not** change the existing TP/FP/FN counts or CSV columns. This is an additional layer on top.

1. **Per-flag, in place.** In `analysis_results/<model>/per_file/<file_id>.json`, add to each `matched: false` flag:
   - `excerpt` (native language, so it's available going forward);
   - `fp_classification`: the classifier output above, plus `classifier_model`, `taxonomy_version: "fp_v1"`, `classified_at` (ISO timestamp), and `prompt_hash` (hash of `prompt.py` content, so results can be tied to a prompt version).

   Also add `excerpt` to matched flags for consistency.

2. **Flat table.** Write `analysis_results/fp_classification/fp_flags.csv` with one row per FP flag. Columns: `model, language, file_id, file_bucket, dataset_origin_label, model_category, timestamp, excerpt, translation, model_confidence, logprob_derived_confidence, entropy_mean, fp_type, fp_reason, evidence_transcript_quote, fp_exception_matched, related_gt_flag_index, transcript_reliability, classifier_confidence, prompt_hash`.

3. **Aggregates.**
   - `fp_type_breakdown_overall.csv` with columns `model, fp_type, n, pct_of_fp`;
   - `fp_type_breakdown_by_language.csv`;
   - `fp_type_breakdown_by_category.csv` (by model category);
   - `fp_type_entropy.csv` with mean entropy and mean logprob confidence per `fp_type` per model. I want to see whether any FP type is separable by entropy.

   Also compute an **adjusted precision** in `adjusted_precision.csv`, per model at flag level and file level. It treats `GT_MISS` as TP and excludes `DUPLICATE_OR_EXTRA_INSTANCE` from the FP count. Report it next to the original precision and show the raw counts.

4. **Human-review export.** Write `analysis_results/fp_classification/review_sample.csv` containing:
   - all `GT_MISS` flags;
   - all flags with `classifier_confidence == "low"` or `transcript_reliability == "suspect"`;
   - a stratified random sample of 5 per (model × fp_type), with a fixed seed.

   Add empty columns `human_fp_type` and `human_notes`. Add a small `--score-review` mode that reads the filled-in file back and reports agreement between classifier and human, overall and per type.

## Script requirements

- Create a new script, `classify_fps.py`. Don't fold this into `analyze_results.py`.
- It must be resumable: skip flags that already have `fp_classification` with the same `taxonomy_version` and `prompt_hash`, unless `--force` is given.
- Flags:
  - `--models` (comma list, same as `analyze_results.py`);
  - `--languages`;
  - `--limit`;
  - `--dry-run` (classify 3 flags, print the requests and responses, write nothing);
  - `--classifier-model`;
  - `--force`;
  - `--score-review <path>`.
- Log to `logs/classify_fps.log` using `pipeline_logging.py`.
- Round all numbers to 2 decimals, as in the other analysis files.
- Write per-file results incrementally so a crash doesn't lose progress.
- At the end, print a summary table: per model, the count and % of each `fp_type`.
- Add a short section to `README.md` describing the script, the taxonomy, the decision order, and every output file.

## Before running the full job

1. Run `--dry-run` and show me the three requests and responses.
2. Run on one model and one language (for example `e2b_thinking`, Hindi) and show me the breakdown along with 10 example rows across different types.
3. Wait for my go-ahead before running across all models.
