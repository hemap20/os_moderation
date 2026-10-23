# Prompting & Conversation-Design Skills

Two skills built for prompt-engineering work: `prompt-engineering` (how to get good
behavior out of a model — reasoning techniques, prompt structure, sampling config,
iteration discipline) and `conversation-design` (how to make chatbot dialogue
engaging and well-paced — currently a placeholder, pending source material).

## How to keep this updated

This repo is meant to grow every time you learn something new from an experiment.
The workflow:

1. Push this repo to GitHub (or keep it in whatever synced location you use).
2. When you have a new learning, start a Claude conversation, attach or connect the
   relevant file(s), and say something like: *"Here's a new experiment finding:
   [describe it]. Update `prompt-engineering/references/X.md` accordingly."*
3. Claude edits the specific file (not the whole skill) — review the diff before
   committing.
4. Add one line to `CHANGELOG.md` describing what changed and why.
5. Commit.

If you connect a GitHub or Drive integration in a future session, Claude can read
and edit files in this repo directly rather than requiring re-upload each time —
ask about that when you're ready to set it up.

## Structure

```
skills/
  README.md                  ← this file
  CHANGELOG.md                ← running log of updates, most recent first
  prompt-engineering/
    SKILL.md                  ← trigger description + decision guide + core principles
    references/
      reasoning-techniques.md              (CoT, ReAct, Tree of Thoughts)
      core-prompting-techniques.md         (zero/few-shot, system/role/context, step-back, self-consistency, APE, code prompting)
      prompt-content-and-assembly.md       (static/dynamic content, RAG, prompt structure)
      model-configuration.md               (temperature, top-K, top-P)
      best-practices.md                    (simplicity, structured I/O, documentation)
      iteration-and-evaluation-discipline.md  (your own field-tested evaluation lessons)
  conversation-design/
    references/               ← empty, awaiting source material
```

## Using this as an actual "Skill"

These folders follow the `SKILL.md` + `references/` convention used by Claude's
Skills feature. Depending on your environment:
- **Claude.ai / Claude apps** — check Settings → Capabilities for custom skill
  upload support.
- **Claude Code** — drop the relevant skill folder into your project's
  `.claude/skills/` directory.
- **Manual use in any chat** — attach the relevant `SKILL.md` + `references/*.md`
  files at the start of a conversation and ask Claude to use them as reference.

Each `SKILL.md`'s `description` field is written to help Claude recognize when a
task should trigger that skill automatically, if your environment supports
auto-triggering.
