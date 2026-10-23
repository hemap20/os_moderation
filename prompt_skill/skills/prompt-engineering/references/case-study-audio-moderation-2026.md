# Case Study: Open-Source Audio Moderation, Prompt Series v1–v6.1 (Sep–Oct 2026)

A record of what was tried, what happened, and what was decided, so the next
project can start from the end state instead of rediscovering it. The general
lessons are distilled in:
- `recall-first-prompt-design.md`
- `evaluation-pipeline-playbook.md`
- `error-analysis-playbook.md`

## Setup

- **Task:** flag policy violations in call audio from an app where users pay
  experts to talk. Categories: PlatformMove (moving contact off the app),
  SuspiciousActivity (payment for sexual content), Explicit-Flirting.
- **Languages:** Hindi, Tamil, Telugu, Kannada, Malayalam.
- **Models:**
  - Gemma audio variants: e2b and e4b, with and without thinking mode (12b only
    in the baseline);
  - text-only runs on Conformer ASR transcripts;
  - a Gemini flash-lite baseline.
- **Ground truth:** Gemini transcribes each file, then classifies it with the
  **original v1 prompt**. A second Gemini step matches model flags to GT flags.
- **Output per flag:** category, timestamp, native quote, translation,
  justification, confidence, and per-token logprobs, from which logprob-based
  confidence and entropy are computed.
- **Goal, as it became clear:** keep FN as low as possible, and have FPs fall
  below a confidence threshold to be set later. **Use the model's self-reported
  confidence for thresholds.**

## Timeline

| Step | What happened | Outcome / lesson |
|---|---|---|
| Baseline (v1) | Precision-first prompt full of alarm language | Precision 0.67–0.92, recall 0.12–0.82. e4b_nothinking was nearly silent. Gemini: 0.92/0.28 |
| FP classification built | 9-type taxonomy, at flag level | The dry run classified **TPs** (an inverted filter) → added an eligibility assertion |
| Matcher audit #1 | Batched Gemini matching: 13% of files disagreed with isolated runs; 5 files silently dropped by a safety block | Switched to one file per call; matching v2 |
| Dev set | 50 files: stratified, enriched, chosen for disagreement, frozen | Iterations took minutes instead of hours |
| v2 | Removed the penalty clause and repeated "do not flag" lines | e2b_nothinking: FP flags 99 → 55; e2b_thinking slightly worse |
| Timestamp bug | `0s`-style timestamps skipped the chunk offset (12–24% of flags) | Defensive normalization |
| v3 | Recall-first framing, one anchored scale, grading instead of filtering | e2b_nothinking: TP 26 → 37, AUROC 0.91; e4b_nothinking: TP 0 → 22 |
| Missing-field analysis | ~10% of `c` missing (no constrained decoding); 95% of those were FPs | Rank missing scores lowest; don't drop them |
| v4 | Speech-act, quote-type and verdict fields, with the verdict placed last | Best ceiling: e2b_nothinking TP 51, FN files 0; P(yes) AUROC 0.93 / 0.85; **but a flood of 1,445 flags** |
| Thinking-offset bug | Logprob lookup ignored the thinking text → all thinking-model logprob and entropy values in v1–v4 invalid | Fixed for future runs; old values treated as missing |
| v5 → "v5*" | Anti-flood rule plus definition fixes, **but run with the wrong schema** (not in the prompt→schema map) | Run largely discarded; the loader now fails loudly; per-field compliance check |
| Threshold tables | File level per category × threshold; TN only at file level; strict (matched) TP | Kept the equivalence assertion against the confusion table |
| Hard-negative audit | 14 confident FPs on clean files | **0 GT misses**; safety advice / reported speech (2 files), fabrication (3), negation (1) |
| Justification mining | 3,767 FP justifications | Hedging rose to 27% (harmless); 51 internal contradictions; "direct" high-confidence FPs dominate |
| Matcher audit #2 | 25 "matcher misses": the native quote was often in the **wrong script** | Translation fallback recovered only 7; the real flaw was matching pairs one at a time |
| Matcher v3 | Global one-to-one assignment plus a content floor plus a verifier | 28 audited pairs validated; stable on repeat runs; TP 155 → 131 (wrong pairs removed) |
| Verifier audit | 25% of rejections at score ≥ 0.9; a mix of errors and correct rejections | No auto-accept band; targeted question/statement instruction; stronger model |
| v6 | Context and speaker fields, a safety-advice rule, negation, contrast pairs, examples per confidence level, "quotes must be real" | e2b_nothinking better (AUROC 0.86, far fewer high-confidence FPs); **e4b_nothinking worse on recall (FN files 7 → 12)**; the context field was ignored (it copied the quote) |
| Speaker × speech act | "reported"/"denial" 0% TP; speaker uninformative | These labels safely remove ~25–40% of FPs; the rest need the score |
| v6.1 | A consolidation pass: removed contradictions, one place per item, 12k → 7.6k characters | Pending results at the time of writing |

## Key numbers to remember

- **Recall-first prompting:** file-level FN went from 5 to 0 (e2b_nothinking), and
  from 25 to 7 (e4b_nothinking, v4).
- **Ranking:** the anchored confidence scale gave AUROC 0.86–0.91 for no-thinking
  models; the verdict-token P(yes) gave 0.93 / 0.85.
- **Thinking models:** self-reported confidence AUROC about 0.60–0.71.
- **Scoring artefacts:** about 44% of early FPs. Of the matcher-related slice, 71
  were duplicates and only 25 were matcher misses (most of those turned out not to
  be).
- **e2b_nothinking under v6:** 0 FN files, but flag recall 0.67 and 1,063 flags in
  50 files.

## Decisions and their reasons

- **Thresholds set only on the full dataset,** per model, from the threshold
  tables, using self-reported confidence.
- **Duplicates are counted as "redundant", not as FPs,** at flag level only; file
  level is unaffected.
- **The matcher is locked (v3)** and every version is re-scored with it.
- **Matching uses a strong verifier model;** no score band is accepted without
  checking.
- **The best prompt is chosen per model:** v6 for e2b_nothinking, v4 for
  e4b_nothinking, pending v6.1 and the thinking-model results. Choose the model
  first, then its prompt.

## Open issues at handover

1. Lock matcher v3 after the verifier fix and spot-checks; re-score v4, v6 and
   v6.1.
2. v6.1 results (no-thinking models) and v6 thinking-model results.
3. Validate the chosen prompt on the full dataset, then set thresholds.
4. A human-labelled dev and test set (the ground truth is Gemini under v1).
5. A grounding check (quote vs Conformer transcript, using the translation too)
   to catch fabrication.
6. A model writing in the wrong script, and models defaulting to 00:00
   timestamps (known prompt-side issues).

## Roadmap beyond prompting (as assessed at the end of the series)

1. **Labels:** a human-labelled dev set and a held-out test set; decide the
   edge-case policies (reported speech, negation, compliments).
2. **Improvements at inference time, without training:**
   - a verification pass per candidate (quote plus ±30 s of transcript → yes/no
     with a probability);
   - self-consistency (several runs, scored by agreement), which mainly helps
     thinking models;
   - overlapping or longer chunks;
   - combining models (high recall from one, precision from another).
3. **Deterministic detectors:** pattern rules for phone numbers, digit groups, UPI
   IDs and handles; spotting app names in the audio directly.
4. **Conversation structure:** speaker diarization; detecting the expert's scripted
   safety advice by comparing against known examples; reconciling flags across
   chunks.
5. **Calibration and thresholds:** isotonic or Platt calibration per model and
   category; conformal prediction for a guaranteed recall level.
6. **Retrieval-based few-shot examples** from a labelled bank of examples.
7. **Training** (after labels exist):
   - a probe on frozen model embeddings;
   - transcribe-then-classify (Indic ASR + MuRIL/IndicBERT or similar);
   - LoRA fine-tuning;
   - preference tuning (DPO) on correct-vs-mistake pairs;
   - distillation from a stronger model;
   - active learning.
8. **External scoring services, for example Jev** (thejevai.com): a hosted
   decision API returning a choice, a score, or a "Noul" yes-probability. It's
   **text-only** (no audio), so it fits as a verifier on transcript windows.
   - Check data handling: transcripts contain sexual content and contact details,
     and the docs describe the operator as separate from the underlying model
     provider.
   - Its Indian-language performance is unknown.
   - Validate its probabilities like any other score.
9. **Other models to test for Indian languages** (check current support for the
   South Indian languages specifically):
   - audio: Sarvam Shuka, Krutrim Dhwani, Qwen3-Omni / Qwen3.5-Omni;
   - ASR: AI4Bharat IndicASR/IndicWhisper, Sarvam Saarika/Saaras (with
     diarization);
   - text: Sarvam-30B/105B (Apache 2.0) as a self-hosted verifier with logprobs.

   Test perception first on the files the current model completely misses.

**Priorities, by the problem found:**

| Problem | Fix |
|---|---|
| Fabricated contact details | Pattern rules plus a grounding check |
| Safety advice taken as a violation | Diarization plus detecting the scripted advice |
| Spoken digits or spelled IDs missed | Pattern rules on transcripts |
| Weak thinking-model confidence | Self-consistency or verification |
| Setting thresholds with low FN | Calibration plus conformal prediction |
| Perception misses | Chunking, then a different or fine-tuned model |
