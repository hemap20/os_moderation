# Changelog

All notable updates to these skills are logged here, most recent first. When asking
Claude to update a skill with a new learning, also ask it to add a line here.

## 2026-10-07
- Added four references from the 2026 audio-moderation series (Gemma audio models
  vs Gemini, 5 Indian languages, prompts v1–v6.1):
  - `recall-first-prompt-design.md`;
  - `evaluation-pipeline-playbook.md`;
  - `error-analysis-playbook.md`;
  - `case-study-audio-moderation-2026.md`.
- Added `prompt-engineering/assets/moderation-eval-kit/`: ready-to-paste task
  prompts plus a template for a recall-first prompt and schema.
- Revised two bullets in `iteration-and-evaluation-discipline.md` (per-category
  confidence calibration; "prefer logprobs over self-report").
- Correction to an earlier draft that was never committed: the claim that
  thinking-model logprobs were "below chance" came from a token-offset bug. It's
  invalid, not a finding.

## 2026-09-23
- Added `prompt-engineering/references/iteration-and-evaluation-discipline.md` —
  field-tested evaluation/iteration lessons from audio content-moderation
  experiments (confidence score validation, mandatory-step cost accounting,
  subtractive vs additive changes, domain transfer pitfalls, error-type diagnosis).
- Added `prompt-engineering/references/best-practices.md`,
  `core-prompting-techniques.md`, `model-configuration.md` — distilled from
  Google's "Prompt Engineering" whitepaper (Feb 2025): zero/few-shot, system/role/
  context prompting, step-back prompting, self-consistency, APE, code prompting,
  sampling parameters, structured I/O, documentation discipline.
- Added Tree of Thoughts section to
  `prompt-engineering/references/reasoning-techniques.md` (Yao et al. 2023).
- Added `prompt-engineering/references/prompt-content-and-assembly.md` — distilled
  from a prompt-engineering book excerpt (content sourcing, RAG, prompt
  structure/assembly).
- Initial `prompt-engineering` skill created: `SKILL.md` +
  `references/reasoning-techniques.md` covering Chain-of-Thought (Wei et al. 2022)
  and ReAct (Yao et al. 2023).
- `conversation-design` skill folder scaffolded (empty — pending source material).
