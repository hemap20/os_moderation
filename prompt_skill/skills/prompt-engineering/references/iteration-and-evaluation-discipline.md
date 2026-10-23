# Prompt Iteration & Evaluation Discipline (Classification/Moderation Tasks)

Field-tested lessons from an extended series of controlled prompt-iteration
experiments on LLM-based content classification (audio moderation). Unlike the other
references in this skill, this file is about *process and evaluation discipline*
around iterating on a prompt over time, not a single technique — most relevant once
you're past initial prompt drafting and into measured refinement, especially for
classification/flagging-style tasks where precision and recall are both live
concerns.

## Confidence scores need validation, not trust

- Self-reported confidence ("rate your confidence 0-1") can carry **zero real
  correlation with correctness**. In one dataset, true positives and outright
  category-misclassifications had *identical* confidence distributions — the score
  tracked "did I detect something" but not "did I get it right." Never treat a
  self-reported confidence field as meaningful until you've checked whether it
  actually separates known-correct from known-incorrect outputs on labeled data.
- Confidence scores cluster near the extremes (very high when confident, well below
  0.5 when not) and rarely land in a useful "medium" band unless the prompt defines
  one. *Revised (2026 series):* per-category calibration rules ("<0.5 for mild cases,
  0.7+ only for severe cases") made confidence reflect the category rule rather than
  correctness. One anchored scale shared by all categories, tied to the definition
  tiers, ranked far better (AUROC 0.75 → 0.91); see `recall-first-prompt-design.md`.
- If the model exposes token-level log probabilities, compute logprob-based scores
  and validate them alongside the self-reported number. *Revised (2026 series):*
  neither wins by default. The logprob of the category token saturates, because the
  model has already committed. A verdict token placed after the reasoning fields
  was the best signal for no-thinking models (AUROC 0.93), while self-reported
  confidence beat the category-token logprob. For thinking models, map positions
  only after the thinking text, or every value is wrong. See
  `recall-first-prompt-design.md` §5.
- Bucketed confidence analysis can silently drop out-of-range values (e.g. a flag
  scoring below your "low" threshold cutoff), making it invisible to bucketed
  reporting even though it still counts toward raw precision/accuracy. Check for
  out-of-range values before drawing conclusions from bucketed confidence data.

## Explicit required fields beat prose instructions to "consider" something

- An instruction to silently reason about something ("consider whether this could
  belong to another category") is much weaker than requiring the model to commit to
  an explicit field capturing that judgment (e.g. a required
  `category_certainty: "certain" | "uncertain"` field). A required field is a real
  behavioral checkpoint; a prose instruction to "think about" something is easy to
  skip past.
- **But** this pattern only pays off if the failure mode it targets is actually
  common. A mandatory field built to catch category-confusion cost measurable recall
  in one case where genuine category-confusion was ~1% of cases — the mechanism
  almost never activated, so it was pure cost. Confirm the target failure pattern is
  real and non-trivial in frequency *before* building a mechanism to catch it.
- Every additional mandatory reasoning step or field has a real, measurable baseline
  cost — even a content-free, tautological step (e.g. asking the model to simply
  restate its own already-made decision, with no new judgment criteria) measurably
  reduced recall in a controlled test. Before adopting a new mandatory step, consider
  testing a content-free control of the same structural shape to measure the generic
  "one more thing to do" tax, and require a real mechanism's benefit to clearly
  exceed that baseline before adopting it.
- For any mandatory gating/checklist mechanism you add, explicitly measure how often
  it actually fires on real cases — not just whether the aggregate metric moved. A
  mechanism that never activates can't be responsible for an observed change; report
  activation rate alongside outcome metrics, or you risk crediting the wrong cause.

## Subtractive changes are underused

- Redundant warnings, emotionally escalated language ("you will be heavily
  penalized," "this is a critical failure"), and repeated restatements of the same
  rule in multiple places tend to become pure cost once basic safeguards are in
  place — not additional benefit. In one iteration series, the single largest
  measured improvement across the whole series came from *removing* a redundant,
  fear-based instruction block and consolidating five scattered repetitions of one
  rule into a single clear statement.
- Escalated/threat-based phrasing is not obviously more effective than plain, calm
  phrasing, and can suppress correct (true-positive) behavior right along with false
  positives — sometimes producing a model that barely flags anything (near-zero
  volume) instead of a well-calibrated one. Pushed too far, this framing is a blunt
  instrument, not a dial.
- Periodically test *removing* things, not just adding — consolidate duplicated
  instructions and check whether performance holds or improves. Don't assume more
  repetition = more effective emphasis.
- Small wording changes (a paragraph removed, a phrase consolidated) can produce
  larger measured improvements than structurally elaborate additions (new fields,
  new gating logic). Impact should be measured empirically, not estimated from how
  sophisticated a change looks on paper — test both, let results decide.

## Exception lists and "when in doubt" instructions have real limits

- Adding a specific example to a "do not flag" list improves the *rate* of correct
  behavior but doesn't guarantee it — a real case nearly word-for-word matching an
  explicitly listed exception can still trigger a false flag. This is ordinary
  probabilistic instruction-following imperfection, not a sign something else is
  broken. Validate aggregate improvement across a real sample; don't expect (or
  require) a single motivating case to be fixed with certainty.
- Concrete positive/negative example pairs embedded directly in a category
  definition ("flag this," "never flag this") are more reliable anchors than
  abstract rule statements alone.
- A single, well-scoped new exception can move accuracy meaningfully on its own —
  narrow, targeted additions are easier to evaluate and reason about than sweeping
  rewrites. Prefer one clear addition you can measure over several bundled changes.
- "When in doubt, do not flag" does not fully eliminate false positives, and
  different models respond to the same instruction with very different aggressiveness
  — one model may still catch most true positives while another goes nearly silent.
  Model choice matters as much as prompt wording for how conservative output ends up.
- A structured multi-step decision process (state rule → check exceptions → check
  context → decide → assign confidence) generally helps precision, but expect
  marginal, sometimes noisy deltas per added step, not dramatic jumps — validate any
  single tweak against a real sample size before trusting it (a 1–2 point accuracy
  change on a 40–80 row sample is often just noise).

## Confident output isn't always grounded output

- A model can assign high confidence and a "verbatim" label to a quote it partially
  or wholly fabricated. Requiring an explicit verbatim-vs-paraphrased label, with a
  hard confidence cap when the model admits paraphrasing, meaningfully reduces (but
  doesn't eliminate) confidently-stated fabrication.
- Fabrication that survives a verbatim/grounding self-check tends to involve the
  model being genuinely, incorrectly *sure* of itself — a harder problem than
  ordinary hedged uncertainty. If false, confident fabrication persists past this
  kind of gate, the fix is more likely architectural than a further prompting patch.
- **Architectural fix**: for any task combining perception (audio/image) with
  judgment (classification/scoring), separate the perception step from the judgment
  step. Producing a full literal transcription first (with no awareness of policy
  categories), then classifying from the resulting text alone, removes the
  classification step's ability to "reason backward" from a conclusion to invented
  supporting evidence, because it never touches the raw signal directly. This also
  produces cleaner ground truth, since a combined perceive+judge call is
  structurally biased toward "finding something to flag" while it transcribes.

## Domain/context framing has outsized effect — more than most wording tweaks

- The conversational/platform context you give the model (what platform this is, who
  the participants are, what a given term typically means in this context) affects
  results more than most tweaks to the violation definition's wording itself. The
  same category definition, framed for the wrong platform, can swing a model from
  strong recall to near-total silence or vice versa.
- Prompts are not very portable across domains even when the underlying policy
  concept is identical (e.g. "don't let users move off-platform"). Never reuse a
  prompt across apps/platforms as a quick test and treat the result as meaningful
  without flagging the domain mismatch — that setup measures
  instruction-following-under-mismatch, not real-world detection quality.
- A prompt change validated on one dataset/platform does not safely transfer to
  another, even when the change seems like a general improvement (e.g. "remove
  excessive caution language"). A subtractive change that improved one platform's
  recall caused a sharp precision collapse on a second platform whose dominant
  failure mode depended more on the caution language that got removed. If aiming for
  one unified prompt across domains, treat unification as its own experiment track
  with its own validation — don't just adopt whichever version wins on your primary
  domain.

## Evaluation-methodology pitfalls (easy to get wrong, invalidate results silently)

- Precision is meaningless (or trivially 0%/undefined) on any sample with zero true
  positives — check the positive rate in ground truth before trusting a precision
  number computed on it.
- Precision/recall computed on very small counts (single digits, especially n≤2–3)
  swing wildly and shouldn't be treated as real signal. Always report the underlying
  tp/fp counts alongside any percentage, and discount cells with tiny denominators
  when comparing prompt versions or models.
- Ground truth quality is itself fallible, not an infallible oracle. When two
  independent experiment models agree on a flag the ground-truth model missed,
  that's a signal the ground truth might be wrong — not that both experiment models
  hallucinated identically. Treat ground truth as a strong baseline to be
  spot-checked on edge cases, not an unquestionable reference.
- Model tier/version naming does not predict performance on a narrow classification
  task — a nominally "older" or "lite" model outperforming a newer one is common
  enough that you should always empirically compare rather than assume newer/bigger
  is strictly better for your specific task.
- A negative result from an automated prompt-optimization tool can be genuinely
  informative rather than a dead end — if a text-only optimizer, tested against known
  failures re-run as text-only inputs, handles all of them correctly, that's evidence
  the failures live in something specific to the original modality (e.g. audio) that
  a text-only tool structurally cannot reach — not evidence the tool is ineffective.
  Check whether a failed optimization attempt is failing because the target problem
  is outside the layer the tool can influence, before concluding the tool doesn't work.

## Distinguish error types before designing a fix

"The model got this wrong" can mean structurally different things needing different
remedies — misdiagnosing the type wastes an iteration on a fix that shows no effect:

- **Fabrication/hallucination** (cited content never actually present) → fix via
  grounding architecture: two-pass perception/judgment separation, verbatim gating.
- **Misclassification** (real content correctly identified, wrong category assigned)
  → fix via clearer category definitions/boundaries.
- **Meta/hypothetical/third-party confusion** (content correctly identified and
  categorized in isolation, but it describes a general practice, hypothetical, or
  someone else's situation rather than an active instance) → fix via explicit
  exceptions distinguishing "described" from "occurring now."
- **Instance-selection variance** (real content genuinely present and correctly
  caught, but a different specific instance than a reference label expected, in
  content with multiple qualifying instances) → this is a measurement/evaluation
  artifact, not a model error — fix the evaluation methodology (credit any correct
  instance per category, not only an exact reference match), not the prompt.

## Practical takeaway
Treat prompt iteration as an empirical discipline, not a craft of "sounds more
careful must be better." Specifically: validate every self-reported signal (don't
trust it because it looks sensible), measure the baseline cost of any new mandatory
step before crediting its benefit, test subtractive changes as seriously as additive
ones, never assume a validated change transfers across domains, and rigorously
separate "the model is wrong" from "the evaluation methodology is wrong" before
spending an iteration on a fix.
