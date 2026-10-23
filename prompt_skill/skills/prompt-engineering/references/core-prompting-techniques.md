# Core Prompting Techniques

Distilled from Google's "Prompt Engineering" whitepaper (Feb 2025). These are the
foundational techniques to reach for before escalating to heavier reasoning
frameworks (CoT/ReAct/ToT — see `reasoning-techniques.md`).

## Zero-shot, one-shot, few-shot
- **Zero-shot**: task description + input, no examples. Try this first — it's the
  cheapest option and works fine for tasks the model already handles well.
- **One-shot**: a single example for the model to imitate. Useful mainly to pin down
  output *format* when zero-shot gets the content right but the shape wrong.
- **Few-shot**: multiple examples showing a pattern. Rule of thumb: start with
  3–5 examples; more for complex tasks, fewer if context length is tight. For
  classification tasks specifically, **shuffle the class order across your examples**
  — if every "positive" example happens to come first in your few-shot set, the model
  can overfit to positional order rather than learning the actual distinguishing
  features of each class. A reasonable starting point is ~6 examples, then tune from
  there based on measured accuracy.
- Example quality matters more than quantity: examples should be diverse,
  high-quality, and representative — including edge cases if you need the model to
  be robust to unusual inputs. (See `prompt-content-and-assembly.md` for more on
  example-selection pitfalls like anchoring bias.)

## System, contextual, and role prompting
Three distinct levers that are easy to conflate — they overlap in practice but serve
different purposes:

- **System prompting**: sets the overall task/purpose ("classify this review",
  "translate this text"). This is the model's fundamental job definition for the
  interaction. Also the right place to put output-format constraints (e.g. "return
  only the label in uppercase") and baseline safety/tone instructions (e.g. "be
  respectful in your answer").
- **Contextual prompting**: supplies situational information specific to *this*
  request (e.g. "you are writing for a blog about retro arcade games") — dynamic,
  changes per-call, helps the model interpret the specific ask correctly rather than
  guessing at missing context.
- **Role prompting**: assigns a persona/identity (travel guide, kindergarten teacher,
  motivational speaker). This shapes tone, style, and the kind of implicit knowledge
  the model foregrounds — e.g. asking for a "humorous" or "inspirational" role-voice
  measurably changes word choice and content emphasis, not just surface style.

Treating these as separate levers (even though a single prompt often uses all three
at once) makes it easier to debug: if output has the right content but wrong tone,
that's a role-prompting fix; if it's missing situational nuance, that's a
contextual-prompting fix; if the model doesn't understand the task at all, that's a
system-prompting fix.

## Step-back prompting
Instead of asking the model to solve the specific task directly, first ask a more
general question that activates relevant background knowledge, then feed *that*
answer back in as context for the specific task.

Example shape: rather than "write a storyline for an FPS level" (which tends to
produce generic output), first ask "what are 5 classic settings that make for a
compelling FPS level?", then follow up with "using [setting X from that list], write
the storyline." The intermediate general-knowledge step measurably improves the
specificity and coherence of the final output versus asking directly.

**When to use it:** tasks where direct prompting tends to produce generic,
underspecified, or biased output because the model jumped straight to specifics
without grounding in general principles first. It's a lightweight technique — much
cheaper than CoT/ToT — worth trying before reaching for heavier reasoning scaffolding.

## Self-consistency
An extension of CoT for tasks where a single reasoning chain isn't reliable enough:
1. Run the *same* CoT prompt multiple times at a **high temperature** (to encourage
   genuinely different reasoning paths, not just token-level noise on one path).
2. Extract the final answer from each independent run.
3. Return the most common answer (majority vote).

This gives a pseudo-confidence signal (how often did the majority answer occur) at
the cost of running the same prompt N times — expensive, but useful when a task is
ambiguous or adversarial enough that single-pass CoT is unreliable (the paper's
example: classifying a sarcastic/socially-engineered email as IMPORTANT or NOT,
where a single CoT pass can land on either answer depending on which framing it
picks up first).

**Note the temperature tension:** this technique needs *high* temperature to get
diverse reasoning paths, which directly contradicts the general CoT best practice of
using temperature 0 for single-pass reasoning (see `reasoning-techniques.md`). Use
temperature 0 when you want one reliable deterministic chain; switch to high
temperature + majority vote specifically when you're doing self-consistency and can
afford the extra calls.

## Automatic Prompt Engineering (APE)
Use the model to generate and refine your prompts, rather than hand-writing every
variant:
1. Prompt the model to generate several semantically-equivalent variants of an
   instruction/query (e.g. "generate 10 different ways a customer might phrase this
   order").
2. Score the candidates against a chosen metric (e.g. BLEU/ROUGE against a reference,
   or a task-specific eval).
3. Keep the highest-scoring candidate as your production prompt (or as one
   representative phrasing among several to train/test against).

Most useful when you need to cover the *range* of ways real users might phrase a
request (e.g. training or testing a chatbot's intent-matching), not just find one
good prompt — the point is breadth of coverage, not a single optimal phrasing.

## Code prompting
LLMs are useful for code-adjacent prompts beyond just "write me a function":
- **Writing code**: works well for well-scoped, common tasks (scripts, boilerplate).
  Always read and test generated code — the model can't verify correctness, only
  pattern-match to what it's seen, and it will confidently produce plausible-looking
  but broken code.
- **Explaining code**: strip comments and ask the model to explain unfamiliar code —
  useful for onboarding onto unfamiliar codebases or reviewing a teammate's PR.
- **Translating code**: converting between languages (e.g. Bash → Python) works
  reasonably well for straightforward, idiomatic code but still needs verification.
- **Debugging and reviewing**: pasting an error traceback *plus* the code and asking
  the model to both fix the immediate bug and suggest broader improvements tends to
  surface issues beyond just the reported error (e.g. missing error handling, edge
  cases like filenames with spaces) — worth asking for both the fix and a general
  review in the same prompt rather than two separate prompts.

## Practical takeaway
Reach for these techniques roughly in this order of increasing cost/complexity:
zero-shot → few-shot → system/context/role framing → step-back → self-consistency.
Only escalate past few-shot + good role/system/context framing when you have
evidence the simpler approach isn't hitting the accuracy/format you need — most
production prompting problems are solved by the cheap end of this list, not by
jumping straight to CoT/ToT/self-consistency.
