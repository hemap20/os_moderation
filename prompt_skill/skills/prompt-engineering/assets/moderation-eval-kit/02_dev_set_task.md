# Task: Build a fixed 50-file dev set for fast prompt iteration

## Why

Prompt experiments on the full dataset (~380 files × several models) take too long, so I want a small, **fixed** dev set of 50 files to iterate on. It must not be random. A random 50 would contain few hard cases and few of each violation category, so prompt changes would barely register. The dev set should be stratified and enriched with the cases prompt changes actually affect. The full dataset stays the final validation set.

Read `dataset.py`, `dataset_v2.py`, `config.py`, `analyze_results.py` and the stage 1/2 ground-truth layout before writing anything, and reuse the existing loaders.

## Dependency

Build the selection from the **re-scored (v2 matching) results** in `analysis_results/`, not from `analysis_results_v1_batched/`. If v2 re-scoring hasn't finished for all 8 models, stop and tell me instead of using v1.

## Signals to compute per file (all 381)

- `language`;
- `dataset_origin_label`, the folder the file sits in (`<lang>_<cat>/TP|FP|FN` or `TN`), and `origin_category`;
- `gt_positive` (ground truth has at least one flag), `gt_categories`, and `gt_flag_count`;
- `n_models_flagged`: how many of the 8 models raised any flag;
- `n_models_correct`: how many models got the file-level bucket right (TP or TN);
- `n_models_fp_flags`: the total count of unmatched model flags across models;
- `duration_s`, `incomplete_transcript`, and any matching-reliability or UNVERIFIED marker from v2 scoring.

## Exclusions

Exclude files that are:
- `incomplete_transcript: true`;
- marked UNVERIFIED or low matching reliability in v2 scoring;
- missing results for any of the 8 models;
- missing the audio file.

Also exclude files longer than the 90th percentile duration, because they dominate run time.

## Selection: 10 files per language × 5 languages

Within each language, pick the following. Break ties with a fixed seed (`--seed 42`). Where more candidates exist than slots, prefer files with the **most model disagreement** (`n_models_correct` closest to 4 of 8). Disagreement files are the ones a prompt change can move.

| Slot | Count | Definition | Purpose |
|---|---|---|---|
| Positive: PlatformMove | 1 | GT has PlatformMove | category coverage |
| Positive: SuspiciousActivity | 1 | GT has SuspiciousActivity | category coverage |
| Positive: Explicit-Flirting | 1 | GT has Explicit-Flirting | category coverage |
| Positive: hard miss | 2 | GT positive, `n_models_correct <= 3` (most models missed it) | measures recall gains |
| Negative: hard clean | 3 | GT negative, `n_models_flagged >= 3` (a clean file several models wrongly flagged) | measures over-sensitivity |
| Negative: easy clean | 2 | GT negative, `n_models_flagged <= 1` | catches a prompt that starts flagging everything |

That gives 5 positive and 5 negative files per language, 50 in total.

If a slot has no candidates in a language, fill it from the closest slot in the same language (positive→positive, negative→negative), and log the substitution. For example, SuspiciousActivity is sparse in some languages. Do not borrow from other languages.

After selection, check these conditions and report whether each holds:
- each category has at least 5 positive files overall;
- there are at least 30 GT flags in total, which is needed for flag-level metrics to mean anything;
- at least 10 of the positives contain more than one GT flag (Multi).

If any fail, adjust within the same language and tell me what you changed.

## Output

1. **Copy the files.** Create `Dostt_dev/` with **the same folder structure as `Dostt/`**, so the existing loaders work unchanged: `<lang>_<cat>/TP|FP|FN/...` and `TN/`, including `Single/`/`Multi/` and the `transcripts/` subfolders. Copy the audio files and every associated ground-truth artifact (transcripts, classifications, raw responses). Copy, don't move. `Dostt/` must be untouched.
2. **Add a `--dataset-root` option** to the dataset config and loaders, defaulting to the current root, so I can run `gemma_local.py`, `gemini_model_eval.py`, `analyze_results.py` and `classify_fps.py` against `Dostt_dev/` without editing code. Results for dev runs must go to separate directories (for example, `gemma_results_dev/`, `gemini_results_dev/`, `analysis_results_dev/`) so they can never mix with full-dataset results.
3. **Write the manifest** to `manifest/dev_set_v1.csv`, with one row per file: every computed signal, the `slot` it filled, `selection_reason`, and whether it was a substitution. Also write `manifest/dev_set_v1_summary.md` with counts per language × slot, per category, GT flag totals, Multi count, total audio duration, and the validation results above.
4. **Write a dev baseline** to `analysis_results_dev/baseline_v1_prompt/`. Extract the existing v2-scored results of all 8 models for just these 50 files: file-level and flag-level confusion per model, overall and per language, and the FP-type breakdown if it exists. The v1 prompt then has a dev-set baseline without re-running any model.
5. **Freeze the set.** The dev set must not change during prompt iteration. If a rebuild is ever needed, write `dev_set_v2`; never overwrite v1.

## Before copying anything

Show me the candidate counts per language × slot, the proposed 50 with their slot and reason, and the validation results. Wait for my go-ahead, then copy and build the baseline.
