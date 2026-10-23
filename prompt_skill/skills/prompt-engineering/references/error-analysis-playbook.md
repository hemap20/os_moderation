# Error Analysis Playbook: Classifying FPs and FNs

How to find out *why* a flagging model is wrong, so each prompt or system change
targets the right cause. A single FP count hides different causes that need
opposite fixes, and one change can fix one cause while making another worse,
leaving the total flat.

Built from the 2026 audio-moderation series. Pipeline details are in
`evaluation-pipeline-playbook.md`.

---

## 1. Principles

- **Classify at flag level, not file level.** Most FP *flags* sit inside files
  already scored TP: one model had 811 FP flags but only 62 FP files.
- **Rule out measurement artefacts before blaming the model.** In the first
  breakdown, about 44% of FPs were scoring artefacts: duplicates, matcher misses,
  and ground-truth misses.
- **Enforce in code every taxonomy rule that can be checked in code.** An LLM
  classifier will break definitions otherwise.
- **The classifier's judgement is evidence, not truth.** Send the high-impact
  labels to human review, and spot-check the rest.
- **Tag every result with the taxonomy version, prompt hash and matcher version,**
  so distributions can be compared across prompt versions.

## 2. The FP taxonomy (ordered: apply in order, stop at the first match)

| # | Type | Definition | Usual fix |
|---|---|---|---|
| 1 | `GT_MISS` | The model is right; the ground truth missed a real violation | Correct the label (versioned log) |
| 2 | `MATCHER_MISS` | The flag is the **same specific utterance** as a GT flag that is **unmatched**, so the matcher failed to pair them | Fix the matcher; counts as TP |
| 3 | `WRONG_CATEGORY` | The same utterance is flagged in the GT under a different category | Clearer category boundaries |
| 4 | `DUPLICATE_OR_EXTRA_INSTANCE` | Another instance of a violation whose GT flag is **already matched** | Scoring policy (count as redundant) |
| 5 | `FABRICATION` | The quote, or anything like it, appears nowhere in the transcript | Grounding check, entropy, verification |
| 6 | `MISHEARING` | A similar-sounding real utterance exists, but the meaning was distorted | Audio and perception (not the prompt) |
| 7 | `EXCEPTION_IGNORED` | Real and correctly understood, but matches an explicit "never a violation" item | Definition structure; instruction-following |
| 8 | `CONTEXT_CONFUSION` | Real, but reported, hypothetical, negated, a refusal, a joke, or safety advice | Speech-act rule, context field |
| 9 | `OVER_SENSITIVITY` | Real, in context, but below the violation threshold | Positive definitions, boundaries, examples |
| 10 | `UNCLEAR` | Not enough evidence (transcript missing, bad timestamp) | — |

**Code-level rules (mandatory):**
- `DUPLICATE_OR_EXTRA_INSTANCE` requires that `related_gt_flag_index` points at a
  GT flag whose verified matched status is true. Otherwise the label is forced to
  `MATCHER_MISS`.
- `MATCHER_MISS` requires the **same specific utterance**, not the same topic. The
  LLM classifier repeatedly over-applied it on topic grounds: "Give me your number
  on WhatsApp" and "no need for WhatsApp" are different utterances.
- Before classifying, check that **the number of eligible flags equals the sum of
  `flag_fp`.** In one run an inverted filter fed *true positives* to the
  classifier, and its labels still looked plausible.

## 3. Inputs for each classified flag

- **The model flag in full:** category, timestamp, native excerpt (load it from raw
  results if the per-file data drops it), translation, justification, speech act,
  speaker, context, confidence, and logprob scores.
- **The GT transcript:** the whole thing, or ±90 s around the flag, plus a search
  of the full transcript for the key terms.
- **All GT flags** for that file, each with its matched status.
- **The other model flags** in the same file (needed to detect duplicates).
- **The policy definitions,** loaded from the prompt file at runtime, never copied
  in by hand.
- **Language,** and the dataset's original label (clearly named, and never
  confused with the model's outcome bucket).

## 4. Classifier prompt requirements

- Walk through the decision order 1→N explicitly, writing the reasoning field
  **before** the type.
- Quote the exact transcript text relied on, or leave it empty, which is required
  for `FABRICATION`.
- Remember the classifier sees a transcript, not audio. It should set
  `transcript_reliability: suspect` when the transcript looks garbled.
- **Remember the model's native text may be in the wrong script.** Judge
  fabrication and mishearing on the translation as well.
- Output: `reasoning`, `fp_type`, `fp_reason`, `evidence_transcript_quote`,
  `evidence_timestamp`, `fp_exception_matched`, `related_gt_flag_index`,
  `transcript_reliability`, `classifier_confidence`.
- Use a **strong classifier model**, stronger than the one that produced the
  ground truth, so it doesn't share its blind spots. Validate responses against
  the allowed values; retry, or mark `UNCLEAR`.

## 5. Process

1. **Dry run on 3 flags, from different file buckets** (TP-file, FP-file and
   FN-file flags). Print the requests and responses, and write nothing. Check
   **which model's results** were used, and that the flags really are unmatched.
2. **Pilot: one model × one language,** full run. Report the breakdown by type,
   split by file bucket (TP/FP/FN files), with 10 example rows across types.
3. **Spot-check about 20 labels** against transcripts, especially duplicates and
   matcher misses. If about 18 or more of 20 are right, proceed.
4. **Full run only for the versions you'll compare.** It must be resumable (skip
   flags already done with the same taxonomy version and prompt hash).
5. **Human review export:** all `GT_MISS` and `MATCHER_MISS` flags, every flag with
   low classifier confidence or a suspect transcript, plus a stratified sample of 5
   per (model × type) with a fixed seed. Add `human_fp_type` and `human_notes`
   columns, and a mode that scores classifier-vs-human agreement.

**Outputs:**
- the per-flag `fp_classification` stored in the per-file JSON;
- `fp_flags.csv`;
- breakdowns overall, by language and by category;
- mean entropy and confidence per type;
- adjusted precision;
- the review sample.

## 6. FN analysis (separate from FP classification)

For every unmatched GT flag, check what the model output nearby:

| Check | Meaning |
|---|---|
| The file has no model flags at all | Detection or perception failure |
| Nothing within ±20 s | Detection failure for that moment |
| A flag of the same category nearby, unmatched | Instance selection (wrong utterance from the same exchange), a duplicate-rule merge, or a matcher miss |
| A flag of another category nearby | Category confusion |

Then read the missed GT quotes, grouped by category, and look for recurring
patterns.

**What it revealed here:**
- **One model's misses (e2b_nothinking) were 100% "flagged something nearby":**
  instance selection, not detection.
- **Another model's (e4b_nothinking) were 80% "nothing nearby":** real perception
  or detection gaps.
- **The recurring wording gaps:** euphemisms ("show", "open"), named sexual acts,
  offers from the expert side, prices tied to time, digits read out in groups,
  spelled-out IDs, and contrast phrases.
- **The prompt's own duplicate rule** ("flag the same thing only once") merged a
  request and the detail itself into one flag, guaranteeing FNs.
- **Some misses were policy questions,** where the ground truth flagged reported
  speech, negations or compliments. Don't change definitions to chase these; get
  a human decision.

## 7. Hard-negative audit (are the confident FPs model errors or GT gaps?)

1. Take every FP at **≥ 0.6** on clean files that several models flagged (the
   hard negatives), for each model in scope.
2. For each one, read the GT transcript around the flag and tag it: fabrication /
   safety advice or reported speech / negation / GT miss / other.
3. **Confirm fabrications against an independent ASR transcript** (here a
   Conformer). One case "fabricated" by the GT-transcript check turned out to be a
   negation once the independent transcript was read.
4. **Count by file as well as by flag.** 9 of 14 flags came from just 2 files.
   Specificity is measured per file.
5. Compare **excerpt entropy** for fabrications against the TP mean, per model.

**Result here: 0 GT misses.** The causes were safety advice and reported speech
(2 files), fabrication (3 files) and negation (1 file). The model labelled the
scam warning `speech_act = direct`, so it never recognised the framing at all.
Telling it to "respect the label" can't help; it has to read the surrounding
context.

## 8. Mining patterns from the model's own justifications

Collect every FP's justification text and search it for recurring patterns:

- **Hedging phrases** ("not a violation", "relates to", "could be interpreted").
  Here they rose from 8% of FPs (v1) to 27% (v3/v4), averaging 0.19 confidence.
  This is harmless by design under a recall-first prompt.
- **Internal contradictions:** the justification says "not a violation" or
  "advising … not to", but the category, confidence or speech act say otherwise.
  Also flags with an invalid category such as `"Not a violation"`.
  - Before adding a post-processing rule, check whether the verdict token or
    P(yes) already ranks these low.
  - A keyword rule can demote real TPs ("this isn't about coins; it *is* a
    platform move"), so validate any rule on a sample.
  - Forcing invalid categories to the lowest score is safe.
- **"Direct", high-confidence FPs:** split them by whether the file has a GT flag
  of the same category (a likely duplicate or matcher issue: re-audit with the
  code-checked classifier) or no GT flags at all (overreach: sample about 20 for
  transcript review).
- Pool across prompt versions only for exploration. For decisions, use the
  versions actually being compared.

## 9. Using model labels as filters: test it

Cross-tabulate TP rate by `speech_act` × `spk` (and by verdict). Here:

- **"reported" and "denial" were 0% TP** (0 of 156), so pushing them to the bottom
  costs no recall.
- **A missing speech act and an "unclear" speaker were also 0% TP.**
- **Together these safely remove about 25–40% of FPs.** The rest are labelled
  "direct" and can only be separated by the score.
- **"hypothetical" wasn't safe:** 8 of 125 were TPs, because requests were
  phrased as questions.
- **The `violation: no` label wasn't safe as a filter either:** 22 of 51 TPs said
  "no" for one model.

## 10. What to do with the results

| Dominant type | Next step |
|---|---|
| Duplicates | Scoring policy: count them as redundant. Leave the prompt alone. |
| Matcher misses | Matcher redesign (global assignment plus verifier), then re-score. |
| GT misses | Versioned label corrections; human labels. |
| Context confusion or ignored exceptions | Prompt: one speech-act section, a context field, consolidated definitions. |
| Over-sensitivity | Prompt: positive definitions, boundaries, counter-examples. |
| Fabrication or mishearing | Grounding check, entropy, a verification pass, a better audio model. |
| FN from perception | Chunking, a different or fine-tuned audio model. Not the prompt. |
