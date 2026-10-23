---
name: prompt-engineering
description: Use when writing, reviewing, or improving prompts and system prompts for LLM-based tools — including few-shot exemplar design, eliciting step-by-step reasoning, building tool-using/agentic prompts, and general prompt clarity and structure work. Trigger on tasks like "write a system prompt for X", "why isn't my prompt working", "help me design exemplars", or "should this prompt use chain-of-thought / an agent loop".
---

# Prompt Engineering

Core techniques for eliciting better model behavior via prompt design, distilled from
primary research and practitioner experience. This skill covers *how the model is
asked to think/act*; for chatbot dialogue quality, persona, and engagement, see the
sibling `conversation-design` skill — combine both when building a conversational bot.

## Quick decision guide

- **Just starting a new prompt / unsure where to begin?** → Start cheap: zero-shot,
  then few-shot, then system/context/role framing, then step-back prompting. See
  `references/core-prompting-techniques.md`. Only escalate to reasoning scaffolding
  (below) once you have evidence the simple approach isn't hitting the target.
- **Task needs multi-step reasoning + you're using a capable model?** → Use
  chain-of-thought exemplars/instructions. See `references/reasoning-techniques.md`
  (Chain-of-Thought section).
- **CoT is unreliable on an ambiguous/adversarial task?** → Try self-consistency
  (sample many CoT paths at high temperature, majority-vote the answer). See
  `references/core-prompting-techniques.md` (Self-consistency section).
- **Task needs the model to call tools (search, retrieval, APIs, code execution)
  interleaved with reasoning about what to do next?** → Use a ReAct-style
  Thought/Action/Observation loop. See `references/reasoning-techniques.md` (ReAct
  section).
- **Task requires search/planning where an early wrong step derails everything
  (puzzles, constrained search, some planning tasks), and you can afford much higher
  inference cost?** → Consider Tree of Thoughts (explore multiple candidate steps,
  self-evaluate, backtrack). See `references/reasoning-techniques.md` (ToT section).
  This is a heavier escalation from CoT, not a default.
- **Task is simple/lookup-style (single fact, single transform)?** → Don't add
  reasoning scaffolding — it adds cost/latency without accuracy benefit, and on small
  or less-capable models can actively hurt.
- **Output is inconsistent / too random / too repetitive?** → Check sampling
  configuration (temperature, top-K, top-P) before assuming the prompt wording is at
  fault. See `references/model-configuration.md`.
- **Need structured/parseable output (extraction, classification, ranking)?** → Ask
  for JSON/XML output directly rather than prose — see `references/best-practices.md`
  for the tradeoffs (token cost, truncation risk) and mitigations.
- **Writing/assembling the prompt itself — what content to include, how to structure
  it, which format to use?** → See `references/prompt-content-and-assembly.md` for
  static vs. dynamic content, retrieval/RAG basics, and structural patterns (sandwich
  structure, lost-in-the-middle, format choice, snippet design).
- **Iterating on a classification/moderation/flagging-style prompt with real
  precision/recall stakes — validating confidence scores, testing exception lists,
  deciding whether a metric change is real or noise, or debugging why a "clearly
  better" prompt regressed?** → See
  `references/iteration-and-evaluation-discipline.md` for field-tested evaluation
  discipline: confidence-score validation, the cost of added mandatory steps,
  subtractive-change testing, cross-domain transfer pitfalls, and telling apart
  different error types before choosing a fix.
- **Optimizing for "catch everything, filter by score later" (recall-first)?** → See
  `references/recall-first-prompt-design.md`: grading instead of filtering, one
  anchored confidence scale, verdict-token logprobs, which structured fields help,
  definition structure without contradictions (Clear/Possible/Never plus one
  "about a violation" section), anti-flood rules, and grounding.
- **Setting up or auditing an evaluation pipeline** (ground truth, dev set, matcher,
  threshold tables, comparing prompt versions)? → See
  `references/evaluation-pipeline-playbook.md`, plus the ready-to-paste task prompts
  in `assets/moderation-eval-kit/`.
- **Diagnosing why a flagging model is wrong** (FP taxonomy, FN analysis,
  hard-negative audits, mining the model's justifications)? → See
  `references/error-analysis-playbook.md`.
- **Starting a similar moderation project?** → Read
  `references/case-study-audio-moderation-2026.md` first: the timeline, numbers,
  decisions, open issues, and the roadmap beyond prompting (verification,
  detectors, calibration, training, Indian-language models).
- **Building a conversational bot that needs both good dialogue *and* tool use /
  reasoning?** → Combine this skill with `conversation-design`: use ReAct-style
  reasoning under the hood for information-gathering turns, and conversation-design
  principles for how the bot phrases things and keeps the dialogue engaging.

## Core principles (expand on these via references as needed)

1. **Match the technique to the task's actual difficulty**, not habit. Adding
   chain-of-thought to every prompt is not a free win — it's conditional on
   (a) genuine multi-step difficulty, and (b) a model capable enough to reason
   coherently. Test both with and without on a representative sample before
   committing to a pattern.
2. **Show, don't just tell.** Few-shot exemplars that demonstrate the desired
   reasoning/action pattern are more reliable than instructions describing the
   pattern abstractly, especially for less-common formats like Thought/Action loops.
3. **Prompt robustness has limits.** Wording, annotator style, and exemplar order
   matter less than the *presence* of the right structural pattern (e.g. reasoning
   before the answer) — but sloppy exemplars still underperform careful ones. Don't
   assume "prompting is fuzzy so anything works."
4. **Design for failure recovery.** Both reasoning-only and action-only approaches
   have known failure modes (hallucination without grounding; state-tracking loss
   without reasoning). When building anything agentic, explicitly plan for
   loop-detection, fallback strategies, and — where the interface allows it —
   human correction points.
5. **Prompt iteration is an empirical discipline, not a craft of "sounds more
   careful must be better."** Validate self-reported signals (e.g. confidence
   scores) before trusting them, measure the baseline cost of new mandatory steps,
   test subtractive changes as seriously as additive ones, and never assume a
   validated improvement transfers across domains without re-testing.

## References

- `references/reasoning-techniques.md` — Chain-of-Thought, ReAct (reasoning+acting),
  and Tree of Thoughts in depth: when each helps, how to construct exemplars/search
  setups, known failure modes, and combination/escalation strategies.
- `references/core-prompting-techniques.md` — Zero/one/few-shot prompting,
  system/contextual/role prompting, step-back prompting, self-consistency, Automatic
  Prompt Engineering (APE), and code prompting — the cheaper techniques to reach for
  before escalating to heavy reasoning scaffolding.
- `references/prompt-content-and-assembly.md` — What to put in a prompt (static vs.
  dynamic content, few-shot example design, RAG/retrieval, summarization) and how to
  structure it once gathered (position effects, sandwich structure, document format
  choice, snippet design, managing multi-element prompts).
- `references/model-configuration.md` — Temperature, top-K, top-P, and output length:
  what they do, how they interact, sensible starting values, and the
  repetition-loop failure mode.
- `references/best-practices.md` — Cross-cutting practices: simplicity and
  specificity, instructions vs. constraints, structured (JSON/XML) input/output and
  its tradeoffs, and how to document and iterate on prompts systematically.
- `references/iteration-and-evaluation-discipline.md` — Field-tested lessons on
  *evaluating and iterating* on classification/flagging-style prompts: confidence
  score validation, mandatory-step cost accounting, subtractive vs. additive
  changes, exception-list limits, grounding/hallucination architecture, cross-domain
  transfer pitfalls, and evaluation-methodology traps (small-sample noise, fallible
  ground truth, error-type misdiagnosis).
- `references/recall-first-prompt-design.md` — Prompt design for recall-first
  flagging: grading instead of filtering, removing alarm language, anchored
  confidence scales, ranking signals (self-report, verdict-token logprob, entropy;
  thinking-mode caveats), structured fields, definition structure, anti-flood,
  grounding, model-specific behaviour.
- `references/evaluation-pipeline-playbook.md` — End-to-end measurement: ground
  truth provenance and corrections, dev-set construction, run plumbing (schema
  mapping, compliance, timestamps, logprob location), matcher design and
  validation, scoring definitions (strict/loose TP, redundant flags), required
  tables, how to compare versions, and an audit checklist.
- `references/error-analysis-playbook.md` — The FP taxonomy with ordered types and
  code-enforced rules, the classification process and human review, FN analysis,
  hard-negative audits, mining justifications, and testing labels as filters.
- `references/case-study-audio-moderation-2026.md` — Timeline, numbers, decisions,
  open issues and roadmap from the audio-moderation series.
- `assets/moderation-eval-kit/` — Ready-to-paste coding-agent task prompts (dev
  set, plumbing, prompt-version setup, FP classification) and a template for a
  recall-first prompt and schema.

(More references will be added here as additional source material is processed —
each new technique gets its own file, linked from this list.)
