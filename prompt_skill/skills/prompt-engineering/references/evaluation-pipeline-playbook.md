# Evaluation Pipeline Playbook for Flagging and Moderation Prompts

An end-to-end recipe for measuring classification or flagging prompts against
reference labels, built from the 2026 audio-moderation series. In that series,
**bugs and design flaws in measurement were as large as the prompt effects being
measured, several times over.** Build this pipeline, and audit it, before
comparing prompts.

Companion files:
- `recall-first-prompt-design.md`: what to change in the prompt.
- `error-analysis-playbook.md`: how to classify FPs and FNs.
- `case-study-audio-moderation-2026.md`: the original timeline.
- `assets/moderation-eval-kit/`: ready-to-paste task prompts for a coding agent.

---

## 0. Pipeline at a glance

```
audio → [model under test] → raw outputs (JSON + per-token logprobs)
                                  │
       ground truth ─────────────►│ matcher (model flag ↔ GT flag pairing)
       (transcript + GT flags)    │
                                  ▼
                       per-file results (each flag: matched?, paired GT idx & category,
                       scores, labels) → confusion tables, threshold tables,
                       ranking metrics → FP/FN classification → audits
```

Every stage has failed silently at least once. Each section below lists the
failure modes and the check that catches them.

## 1. Ground truth

**Know where it came from.** Here, the ground truth was Gemini running the
**original v1 prompt** on its own transcript. That has consequences:

- **Every metric measures agreement with that prompt's policy.**
  - A prompt change that *clarifies* what the ground truth already does (for
    example, "reported speech isn't a violation") is measured cleanly.
  - A change that *alters the policy* will look like a regression, even when it's
    better.
  - Decide which kind each change is before running it.
- **Don't let prompt changes drift from the ground-truth policy without noticing.**
  By v6, some definitions ("show/open", "mentioning an app") no longer matched
  the v1 policy.
- **Keep the ground truth's schema and prompt separate from the experiment's.**
  Here they shared a schema file, so changing the experiment's schema would
  silently have changed future ground truth. Give each its own module and add an
  assertion that the ground-truth generator uses the original.

**Corrections:**
- Fix confirmed ground-truth errors through a **versioned correction log**
  (`gt_corrections_v1.csv`: file, flag, who checked it, why). Re-score *every*
  version against the corrected labels, and keep the original alongside.
- Verify suspected ground-truth misses against transcripts, and against an
  independent ASR transcript, before correcting. Here, a hard-negative audit found
  **0 ground-truth misses in 14 suspected FPs**; all were model errors. The
  default assumption should be "the model is wrong".

**The lasting fix: a human-labelled dev and test set.** 50 dev files is about a
day of labelling, plus a stratified held-out test set (~150 files) that is never
tuned on.

## 2. The dev set (fast iteration)

- **Stratify and enrich it; don't sample randomly.** The layout used, per
  language (10 files):

  | Slot | Files | Purpose |
  |---|---|---|
  | One positive per category | 3 | Category coverage |
  | Hard positives (most models missed them) | 2 | Measures recall gains |
  | Hard negatives (clean files that ≥ 3 models flagged) | 3 | Measures over-sensitivity |
  | Easy negatives | 2 | Catches a prompt that starts flagging everything |

- **Break ties toward model disagreement** (files where about half the models are
  right). Those are the files a prompt change can move.
- **Exclude** incomplete transcripts, files with unreliable matching, files
  missing results for any model, and very long audio (above the 90th percentile).
- **Validate the set:** at least 5 positives per category, at least 30 GT flags,
  and at least 10 files with multiple GT flags.
- **Mirror the original folder structure** so loaders work unchanged, add a
  `--dataset-root` option, and keep separate result folders per dataset and prompt
  (`*_dev_promptvN/`).
- **Extract the old prompt's dev baseline from existing results.** No re-run is
  needed.
- **Freeze it** (`dev_set_v1`). Use it for direction, not proof: one file is 4
  points of file-level recall. Confirm the final prompt on the full dataset.

## 3. Running models: parsing and logprobs

- **Map each prompt to its schema explicitly, and fail loudly.** Here, a new prompt
  file wasn't in the prompt→schema map and silently fell back to the old 6-field
  schema. The prompt text asked for 9 fields, the embedded schema showed 6, and
  most models followed the schema: 92–94% of flags were missing the verdict field.
  The run had to be discarded.
  - Let each prompt declare its own schema (`SCHEMA = "v6"`).
  - Raise an error for unmapped prompts.
  - At the end of every run, report the missing rate per field, and **warn if any
    required field is missing on more than 20% of flags.**
- **Without constrained decoding, required fields are only suggestions.** Expect
  about 10% omission. Use grammar- or schema-constrained decoding if possible, and
  check it doesn't change what the model writes. Never fill in missing values when
  parsing.
- **Normalize timestamps defensively.** Models wrote `0s`, `5s`, ranges, and
  `H:MM:SS`. The parser logged them but kept the raw value and skipped the chunk
  offset, which broke matching for 12–24% of flags. Accept every format, convert to
  seconds, and add the chunk offset.
- **Chunk-start timestamps** (relative 00:00) are often the model's default when
  unsure, not real times. Flag them as low-reliability.
- **Locating logprobs:**
  - Find the value of the relevant field (for example `"violation"`) *within that
    flag's own JSON object*, and take the logprob of its first token.
  - Store the top-k alternatives at that position, so P(yes) can be read for "no"
    verdicts.
  - **Thinking models:** the token stream includes the thinking text before the
    answer, so slice the tokens to the answer before mapping character positions,
    and handle answers that start partway through a token.
  - Verify on 3 real outputs by printing the located token, its logprob and the
    top alternatives. If the anchor isn't found, set the field to `None` and log
    it; never fall back silently.
- **Record per result:** the prompt hash, the schema version, the matcher version,
  the model, and the chunk length.

## 4. The matcher (pairing model flags with ground-truth flags)

This was the most error-prone component. Lessons, roughly in the order they were
learned:

1. **Batching LLM matching calls is nondeterministic:** about 13% of files
   disagreed between batched and one-file-per-call runs, with some files' results
   dropped or garbled. Use **one file per call, temperature 0, cached**.
2. **Silent drops change the denominator.** A safety block dropped 5 files from one
   model's scoring with no retry. Assert that every model was scored on the same
   set of files.
3. **Native-text similarity fails when the model writes in the wrong script.** Use
   similarity of the English translations too.
4. **String similarity fails on paraphrases and creates false matches from single
   shared words** ("super"). Use multilingual **sentence embeddings** for meaning
   similarity.
5. **Similarity can't tell whether two quotes are the same instance.** Shared
   domain vocabulary ("show", "WhatsApp", "300") pushes scores high: pairs at
   similarity 1.00 were sometimes different utterances ("WhatsApp number" vs "I
   changed my phone number"). No score cut-off separates good from bad pairs.
6. **Matching one pair at a time misses better candidates.** A ground-truth flag
   got paired with a distant flag while closer, near-identical ones went unused.

**The design that worked (matcher v3):**

```
for each file:
  candidates = every (GT flag, model flag) pair within ±60 s
  score(pair) =
      content = max(translation_embedding_sim, native_sim if script-valid else 0)
      time    = soft decay with distance; if model timestamp == chunk start → "unknown
                within chunk" (scored as somewhere in that chunk, not a point)
      category bonus (match) / penalty (mismatch; cross-category still possible)
  hard gate: content ≥ minimum floor (time+category alone can never make a match)
  assignment: optimal one-to-one (Hungarian) over the score matrix, min score to qualify
  verification: LLM (STRONG model) confirms/vetoes each assigned pair, seeing native
                text + translation + timestamps of both sides
                → on veto: drop that pair, re-run assignment (next-best candidate)
  record: score components, verified?, timestamp-unknown?, paired GT index + category
```

**Verifier lessons:**
- **The verifier decides correctness,** because the score can't. Use the strongest
  model available; the lite model was the weak point three times.
- **Don't add a score band that's accepted automatically without checking it
  against the data.** In this data, rejections at ≥ 0.9 were a mix of verifier
  errors and correct rejections at every score level.
- **Fix named verifier biases with targeted instructions.** For example, it
  rejected pairs only because one side was a question and the other a statement,
  which is a translation artefact in these languages. The instruction: "don't
  reject solely on question vs statement phrasing; judge whether it's the same
  moment and the same content". Remember that a question and its answer can be
  two different utterances by different speakers.

**Validating a matcher change:**
- **Build a small audited set of pairs** (about 28 here): each lost or gained pair
  tagged as *correctly declined*, *genuine regression*, *timestamp artefact*,
  *better candidate missed*, *false gain* or *uncertain*.
- **Score the new matcher on that set.** Make only *principled* fixes, ones you'd
  choose without looking at a specific case (for example, max of the two content
  signals, a content floor). Then **stop tuning**: the set is now your validation
  data.
- **Don't let the matcher under test overrule your manual audit.** Mark
  disagreements as uncertain, and settle them by listening to the audio.
- **Run it twice** and report the agreement rate (100% here).
- **Report gross gains and losses, not just net TP.** Net 0 can hide equal
  numbers gained and lost.
- **Raw TP count is the wrong success measure.** It fell from 155 to 131 when wrong
  pairings stopped being counted.
- **Spot-check both rejected and confirmed pairs** (about 20 each; lock the matcher
  if roughly 17 or more of each are right).
- **Lock the matcher version** for the whole prompt series, record it with every
  result, and **re-score every earlier version** whenever it changes.

## 5. Scoring definitions

**File level, for a category C** (or ALL), at threshold t, using the score you
plan to threshold on:

| | Has a matched flag of C (to a GT flag of C) with score ≥ t | Otherwise |
|---|---|---|
| **GT positive for C** | TP | FN |

| | Has any flag of C with score ≥ t | Otherwise |
|---|---|---|
| **GT negative for C** | FP | TN |

- **The TP rule requires a verified match** (strict). Also report
  `loose_tp`/`loose_recall`, where any flag of C at or above t counts. The gap
  between them is "right file, wrong utterance".
- **Recording which GT flag each model flag was paired with** (and its category)
  is necessary for the per-category rule.
- Precision, recall, specificity, accuracy and F1 come from these counts. **TN only
  exists at file level.**

**Flag level, for category C at threshold t:**

| Outcome | Definition |
|---|---|
| TP | Matched flags with score ≥ t |
| **Redundant** | Correct extra instances of an already-matched GT flag (labelled by the FP classifier as duplicates, with the code-level check). Not an error, and not a new catch. |
| FP | All other flags with score ≥ t |
| FN | GT flags − TP |

- **Flag recall** = TP ÷ GT flags. **Flag precision** = (TP + redundant) ÷ flags.
- Report the redundant count separately. In production, de-duplicate per file and
  category before review.
- Duplicates don't affect file-level metrics: they always sit in files the model
  already got right.

**Missing scores:** rank them lowest (below every threshold), and report how many
there are. Never drop them: they were 95% FPs, and excluding them inflated ranking
metrics and hid 2 TPs. Include a **"none" threshold** row that uses every flag.

## 6. Output tables to produce for every prompt version

1. **Confusion tables** at file and flag level, overall and per language.
2. **A threshold-by-category table** (`threshold_metrics_by_category.csv`): model ×
   language × category (including ALL) × threshold (none, 0.0–0.9 in steps of
   0.1), with TP/FP/FN/TN, precision, recall, specificity, accuracy, F1,
   `strict_tp`, `loose_tp`, flag TP/FP/FN/redundant, flag precision and recall,
   and the count of flags with a missing score. **Assertions:**
   - TP + FN = number of GT positives, and FP + TN = number of GT negatives;
   - the "none" row for ALL equals the confusion table;
   - TP and FP never rise, and TN never falls, as the threshold rises.
3. **Per-bucket and cumulative confidence tables:** each bucket's precision, share
   of all TPs and FPs, and recall contribution, plus cumulative "at or above"
   metrics. Use finer edges near 0 and 1 for probability scores.
4. **Ranking metrics per score:** AUROC, AUPRC, and recall at a fixed FP budget
   (10/25/50/100 FPs), overall, per language and per category.
5. **Label tables:** TP rate by speech act, by speaker, and by verdict, plus
   speaker × speech act, and the verdict-vs-matched 2×2. These show whether the
   model's own labels carry information.
6. **FP-type breakdown** (see `error-analysis-playbook.md`), plus adjusted
   precision (ground-truth misses counted as TP, duplicates counted as redundant).
7. **Flags per file** (distribution and maximum), the share of invalid categories,
   the missing rate per field, and the share of quotes written in the expected
   script (via Unicode ranges).

## 7. How to compare two prompt versions

1. **Same dev set, same matcher version, same ground-truth version.** Check all
   three.
2. **Primary:** FN files and flag recall at "none" (the ceiling), then recall at
   the operating thresholds you're considering (for example 0.6 and 0.8).
3. **Secondary:** FP files and specificity at those thresholds, and AUROC of the
   score you'll threshold on.
4. **Context:** flag volume, redundant count, and compliance (missing fields,
   invalid categories).
5. **Explain the difference:** FP-type shifts, speech-act labels on the
   hard-negative files, and per-file differences for files whose bucket changed.
6. **Look per model, and per category.** Effects often go in opposite directions.
7. **Don't set thresholds during iteration.** Set them once, per model (and
   possibly per category), on the full dataset, using the same threshold tables.
   Consider calibration (isotonic regression) and conformal prediction for a
   guaranteed recall level.

## 8. Audit checklist before trusting any comparison

- [ ] Every model was scored on the same files; no silent drops.
- [ ] Prompt→schema mapping verified; per-field missing rate below 20%.
- [ ] Timestamps normalized; chunk-start timestamps flagged.
- [ ] Logprob locator verified on real outputs (thinking models included).
- [ ] Matcher version locked and identical across the versions compared.
- [ ] The "none" row of the threshold table equals the confusion table (assertion).
- [ ] Selection assertions hold (for example, eligible FP flags = the sum of
      `flag_fp`).
- [ ] The judge or verifier model is stronger than the ground-truth generator.
- [ ] The ground-truth correction log is applied to every version, or to none.

## 9. Cost and runtime notes

- Thinking modes were much slower, and their self-reported confidence didn't
  respond to prompt changes. Iterate on the fast models, and re-test the slow ones
  with the final prompt.
- Get a cost estimate before full runs: flags to classify, calls, and retries.
  Restrict expensive classification to the versions that will actually be compared
  (here v4 and v6, not every historical version).
