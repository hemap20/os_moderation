# Recall-First Prompt Design for Classification and Flagging

How to write prompts for flagging and moderation tasks when the goal is **"miss as
little as possible, and let FPs exist only below a score threshold set later"**,
as opposed to precision-first ("only flag when sure").

Evidence comes from an audio-moderation series (Sep–Oct 2026): open-source Gemma
audio models (e2b/e4b, with and without thinking mode) on calls in 5 Indian
languages, with 3 categories (PlatformMove, SuspiciousActivity, Explicit-Flirting),
iterated over prompts v1–v6.1 on a 50-file dev set. See
`case-study-audio-moderation-2026.md` for the timeline and numbers. Figures quoted
here are dev-set results: treat them as directional, not proven.

Companion files:
- `evaluation-pipeline-playbook.md`: how to measure.
- `error-analysis-playbook.md`: how to diagnose FPs and FNs.

---

## 1. Decide the objective first, because it changes the whole prompt

| Objective | What the prompt should do | What to measure |
|---|---|---|
| Precision-first | Suppress doubtful cases ("if ambiguous, don't flag") | Precision and recall at the model's own decision |
| **Recall-first + threshold later** | **Output every plausible candidate and grade it** | Recall with every flag kept (the recall ceiling), plus how well the score ranks TPs above FPs |

**The recall with every flag kept is a hard ceiling.** A threshold can only remove
flags. Every "when in doubt, don't flag" rule removes candidates before they're
scored, and they can never be recovered. Under a recall-first objective,
**suppression rules are the main thing to remove.**

## 2. Grade candidates instead of filtering them

- **Turn filtering rules into grading rules.** Instead of "don't flag reported
  speech", use "output it, label it as reported, and give it low confidence".
- **Keep hard exclusions only for genuine non-violations**, such as features the
  policy explicitly allows (in-app video calls, platform coins). Flagging those
  only adds noise.
- **Tell the model how its output will be used:** "Your confidence will be used to
  filter flags later, so a missed violation is worse than a doubtful flag with low
  confidence." This measurably changed behaviour.

**Observed effect** (precision-first v1 → recall-first v3/v4):
- e2b_nothinking: TP flags 26 → 51 (of 73); FN files 5 → 0.
- e4b_nothinking, which v1 had suppressed completely: TP flags 0 → 33 (4 flags in
  total under v1).
- The extra FPs went almost entirely to the bottom of the scale: in v3, 285 flags
  were at confidence ≤ 0.3, and only 2 of them were TPs.

## 3. Remove alarm and penalty language

- **Penalty clauses, "CRITICAL", "NEVER", and the same "do not flag" repeated six or
  more times all suppress output**, mostly at the expense of recall. Consolidate to
  one calm statement placed at the end of the prompt.
- **The effect differs by model.** Removing alarm language *nearly halved* FP flags
  for e2b_nothinking (99 → 55, with recall held), yet slightly hurt e2b_thinking.
  Test every change on every model you care about.

## 4. Use one anchored confidence scale, never per-category rules

- **Per-category confidence rules destroy ranking.** Rules like "below 0.5 for mild
  flirting", "at least 0.7 to flag other categories" or "paraphrases below 0.5"
  make confidence reflect the category and the rule, not how likely the flag is to
  be correct.
- **Use one scale with anchors, shared across categories.** The version that
  worked:

  | Score | Meaning |
  |---|---|
  | 0.9 | Clear |
  | 0.6 | Possible, or clear but the audio or intent is unclear |
  | 0.3 | About a violation (reported, warning, negation) or a weak match |
  | 0.1 | Related words, probably nothing |

  e2b_nothinking's AUROC rose from 0.75 to 0.91 with this scale plus the
  recall-first framing.
- **Tie the scale to the definition tiers** (Clear → 0.9, Possible → 0.6, About →
  0.3, see §7), so the two can't disagree. In v6, the scale's examples put
  "Are you on Insta?" at 0.6 while the definition listed "Are you on WhatsApp?" as
  a clear case. Inconsistencies like that reduce how reliable the confidence is.

## 5. Which ranking signal to trust depends on the model; validate each one

| Signal | Finding |
|---|---|
| Self-reported confidence | Good for no-thinking models once anchored (AUROC 0.79–0.91). **Uninformative for thinking models** (around 0.60–0.71): they settle the decision while reasoning, then report it as 0.9 whether it's right or wrong. |
| Logprob of the category token | Usually saturated near 1.0. The token is written after the model has committed, so it carries little information. |
| **Logprob of a verdict token** (`violation: yes/no`, placed after the reasoning fields) | The best signal for no-thinking models: AUROC 0.93 (e2b) and 0.85 (e4b). |
| Excerpt entropy | Separated fabrication for e2b_nothinking (0.62–0.82 vs a TP mean of 0.39), but not for e4b_thinking. |

- **Neither self-reports nor logprobs win by default.** This revises the earlier
  "prefer logprobs" rule. Compute both, and check each per model on labelled data.
- **Field order is part of the prompt.** The evidence fields (quote, translation,
  context, speech act, justification) must come *before* the verdict and the
  confidence. State the order in the instructions *and* declare the fields in that
  order in the schema, because the schema is embedded in the prompt.
- **For "no" verdicts, read P(yes) from the top-k alternatives** at that position.
  Fall back to 1 − P(no) only when "yes" isn't in the top-k, and record which
  method was used.
- **Thinking models:** the token stream includes the thinking text before the
  answer. Any lookup that maps character positions to tokens must start where the
  answer starts, or every logprob and entropy value is read from the wrong token.
  (This bug made all thinking-model logprob results in v1–v4 invalid.)

## 6. Structured fields: add the ones the model fills honestly

| Field | Verdict |
|---|---|
| `speech_act` (direct / reported / hypothetical / denial) | **Keep.** "reported" and "denial" were 0% TP across 156 flags, so the label is reliable when used. Its weakness is that it's *under-used*: real warnings got labelled "direct". |
| `violation` (yes / no), placed last | **Keep**, for the verdict-token logprob. The *label* itself can't be used as a filter: e2b_nothinking said "no" to 22 of its 51 TPs, while its P(yes) still ranked them well. |
| `quote_type` (verbatim / paraphrase) | **Drop.** The models label nearly everything "verbatim", so it carries no signal. |
| `spk` (expert / user / unclear) | **Weak.** TP rates were equal for expert and user. "unclear" was 0/53 TP, so it's useful only as a negative marker. |
| `ctx` (the words around the quote) | **Small models ignored the intent:** they copied the quote, in the native script, instead of describing the context. Specify the language, say "do not repeat the quote", and check compliance. |

Every added field also adds output length and another chance to omit a field.
Measure the omission rate per field.

## 7. Definitions: structure them so each item lives in exactly one place

Rules added over several rounds of fixes pile up into contradictions. In v6, by
the end:

- **Safety advice was both "output it at ≤ 0.3"** (in the speech-act section)
  **and "do not output it"** (a "Not a violation" item, governed by the output
  rule). The model's behaviour on it was split.
- **Compliments and relationship talk appeared in both** the "low confidence" list
  and the "not a violation" list.
- **"Flag anything plausible" competed with "do not output when …"**, so the model
  tried to satisfy both: it emitted the flag and gave it the category
  `"Not a violation"` (242 to 783 such flags per run).

**A structure that avoids this (v6.1):**

```
[Output rule]  ONE rule, stated once: output Clear, Possible and About items;
               never output Never items or ordinary conversation.
[Category X]   Definition (one sentence)
               Clear (0.9):    items + examples
               Possible (0.6): items + examples
               Never (don't output): items + examples
[About a violation, not doing it]  ONE section for ALL categories:
               warnings, safety advice, reported speech, refusals, negation,
               questions about the rules → output with violation=no, c ≤ 0.3,
               plus the cue words that signal them ("don't", "they will say", …)
[Confidence scale]  tied to the tiers above
[Fields]       in order; "violation" defined from the other fields
```

**Definition-writing rules learned:**

- **Write positive definitions** (what the violation *is*: "a speaker proposes
  continuing contact outside this app") instead of growing exception lists.
- **Write in-app features into the definition itself** ("every call type inside
  the app is allowed"). A separate exception list didn't stop a no-thinking model
  from flagging in-app video calls.
- **Use the speech-act rule** ("who is doing it, and is it happening now?"). Most
  FPs were speech *about* a violation (warnings, stories, denials, questions about
  the rules), not violations. One rule covers every category.
- **Avoid keyword triggers.** "Suggestive comments", with no boundary, fired on any
  "sex" or "sexy". A broad gloss ("show/open means nudity") fixes misses but needs
  a boundary ("when it's about the body or the video"), with counter-examples.
- **Spell out contact details in every spoken form** (digit groups, letters spelled
  one by one, contrast phrases like "not here, on Telegram") for the
  contact-detail category.
- **Separate utterances are separate flags.** A request ("give me your number") and
  the detail itself are two flags. Only the same quote repeated is a duplicate.
- **Write examples yourself; never paste dev-set quotes into the prompt.** That
  leaks the evaluation data and inflates dev scores.

## 8. Anti-flood: ask for candidates without inviting everything

- **"0.1 = mentioned, probably not a violation" plus a legitimate "no" verdict was
  read as "list everything" by a 2B-effective model:** 1,445 flags in 50 files, up
  to 94 in one file, including greetings.
- **Fix:** "only flag quotes related to a category definition; ordinary
  conversation is never a flag", with an explicit list (greetings, small talk,
  introductions, call quality), and redefine 0.1 as "related to a definition (names
  an external app, money, sex) but probably not a violation".
- **Output caps may apply per chunk, not per file.** Check how many files hit the
  cap, and whether low-confidence flags push real ones out.

## 9. Grounding and fabrication

- **Fabrication at confidence 0.95 does happen,** and no confidence threshold
  removes it. Prompt rules ("quotes must be real, never join words from different
  places, quote what you hear and cap confidence at 0.3 if unsure") help a little.
- **The real fixes are outside the prompt:** fuzzy-match each quote against an
  independent ASR transcript (on the translation too, see below), use entropy where
  it separates, or add a verification pass.
- **Audio models sometimes write the native quote in the wrong script** (Malayalam
  in Bengali letters, Kannada as garbled Latin) while the English translation is
  fine. Any check that relies on the native text has to tolerate this.
- **Models default to "00:00" when unsure of timing** (the chunk start). Ask for
  "best estimate, never 00:00 unless it really is at the start", and make the
  matcher treat chunk-start timestamps as unknown.

## 10. Model-specific behaviour seen

- **Cautious models lose recall with every added caution.** e4b_nothinking: v4
  missed 7 files, v6 missed 12. More context rules made it drop real candidates.
  Compare FN per model for each prompt change.
- **Small models degrade as prompts get longer and denser:** missing fields (about
  10% without constrained decoding), invalid categories, ignored field
  instructions. Consolidating v6 from about 12k to 7.6k characters was itself a
  fix.
- **The best prompt can differ by model** (here v6 for e2b_nothinking, v4 for
  e4b_nothinking). If only one model will be deployed, choose the model first,
  then its best prompt.

## 11. Process lessons

- **One variable per run is ideal,** but bundling is acceptable when the changes
  serve one purpose (for example, switching from filtering to grading). Say so,
  and use FP-type shifts and per-file diffs to work out which part did what.
- **Know when to stop wording changes.** Once file-level recall is at its ceiling
  and the remaining FPs come from fabrication, wrong attribution and perception,
  the next gains come from verification, grounding, labels or training, not more
  rules.
- **Before declaring a "final" prompt, do a consolidation pass** that only removes
  contradictions and duplicate rules, adding nothing new.
