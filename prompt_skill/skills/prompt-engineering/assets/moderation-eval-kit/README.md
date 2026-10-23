# Moderation Evaluation Kit

Task prompts and templates from the 2026 audio-moderation series, ready to paste into
a coding agent (Claude Code) for the next flagging or moderation project. They were
written against a specific repo (`os_moderation`), so **replace the file and module
names, categories and model names with your project's own.** The structure, checks
and order of work carry over.

Read these first:
- `../../references/evaluation-pipeline-playbook.md`
- `../../references/error-analysis-playbook.md`
- `../../references/recall-first-prompt-design.md`

## Files, in the order you'd use them

| File | Use |
|---|---|
| `02_dev_set_task.md` | Build a stratified, enriched, frozen dev set (50 files) plus a free baseline from existing results |
| `03_schema_and_logprob_plumbing_task.md` | Add structured fields and verdict-token logprobs (P(yes) from the top-k), with the ground-truth schema kept separate |
| `04_new_prompt_version_setup_task.md` | Wire in a new prompt version safely: the prompt declares its schema, unmapped prompts raise an error, per-field compliance warning, dry run, then a dev run |
| `01_fp_classification_task.md` | Flag-level FP taxonomy, with code-enforced rules, resumable runs, a human-review export and adjusted precision |
| `example_prompt_recall_first_v6_1.py` | Template for a recall-first prompt: one output rule; Clear/Possible/Never per category; one "about a violation" section; a scale tied to the tiers; field order with the verdict last |
| `example_schema_v6.py` | A schema whose field order matches the prompt, with fields optional so omissions are measured rather than crashing the run |

## Updates to make when reusing the older task prompts

These task prompts were written mid-series, and later lessons change a few details:

- **FP classification (01):**
  - Add the `MATCHER_MISS` type before duplicates, with the code rule (a duplicate
    requires a matched related GT flag), and the "same specific utterance, not the
    same topic" criterion.
  - Use a strong classifier model, and remember the model's native quotes may be in
    the wrong script.
- **Matching:** use the matcher v3 design in the pipeline playbook (global
  one-to-one assignment, a content floor, a strong verifier, chunk-start timestamps
  treated as unknown). Batched pairwise matching is not acceptable.
- **Threshold tables:** use the strict, match-based TP definition with
  `loose_tp`/`loose_recall` alongside, a "none" row that must equal the confusion
  table, and a `redundant` flag outcome.
- **Thinking models:** slice tokens to the answer before locating any logprob.
