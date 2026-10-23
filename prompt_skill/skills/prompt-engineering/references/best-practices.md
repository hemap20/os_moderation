# Prompt Engineering Best Practices

Distilled from Google's "Prompt Engineering" whitepaper (Feb 2025). Cross-cutting
practices that apply regardless of which technique (zero-shot, CoT, ReAct, etc.)
you're using.

## Design with simplicity
If a prompt is confusing to you, it will likely confuse the model too. Prefer short,
direct, plain instructions over elaborate or verbose phrasing. Lead with an action
verb that names exactly what you want (e.g. Classify, Summarize, Extract, Compare,
Rewrite, Translate) rather than burying the ask in a paragraph of scene-setting.

Compare: "I am visiting New York right now with two 3-year-olds, where should we go"
vs. "Act as a travel guide for tourists. Describe great places to visit in Manhattan
with a 3-year-old." The second is more specific about role and constraints while
being no longer — simplicity and specificity aren't in tension.

## Be specific about the desired output
A vague ask forces the model to guess at scope, length, tone, and structure. State
these explicitly: "Generate a 3-paragraph blog post about the top 5 video game
consoles, informative and engaging, conversational style" beats "generate a blog
post about video game consoles." Specificity narrows the model's guessing space and
directly improves accuracy/relevance, not just tone-matching.

## Prefer instructions over constraints
- An **instruction** says what to do ("only discuss the console, company, year, and
  sales figures").
- A **constraint** says what *not* to do ("don't list game titles").

Positive instructions tend to outperform negative constraints: they tell the model
what target to aim for, rather than leaving it to infer the target from an
exclusion list — and a long list of constraints can be internally contradictory in
ways that are hard to spot. Constraints still have a real place — safety/harm
boundaries, or when a strict format must be enforced — but default to stating the
positive goal first, and add constraints only where necessary.

## Control output length deliberately
Two levers, use both when it matters: a hard token limit in the model config, *and*
an explicit length request in the prompt itself ("in one paragraph", "in tweet
length"). Relying on the token limit alone risks a response truncated mid-sentence
rather than a genuinely concise one.

## Use variables in prompts
When a prompt template will be reused with different inputs (e.g. `{city}`,
`{user_name}`), parameterize it explicitly rather than hardcoding one example value
and hand-editing per use. This matters directly once a prompt moves from
experimentation into an actual application/codebase.

## Experiment with input formats and phrasing
The same underlying ask can be phrased as a question, a statement, or an
instruction, and these can produce meaningfully different outputs from the same
model. Don't assume your first phrasing is optimal — when a prompt underperforms,
try reformulating the *type* of phrasing, not just tweaking wording within the same
type.

## Adapt to model updates
Prompts are not portable across model versions/families without re-testing.
Behavior — even from the same vendor across versions — can shift enough that a
previously-working prompt needs adjustment. Re-test existing prompts whenever you
change model or model version, not just when you write a new prompt.

## Prefer structured output formats (JSON/XML) for non-creative tasks
For extraction, classification, parsing, ranking, or any task with an inherent data
shape, ask for structured output rather than prose. Benefits:
- Consistent, parseable shape every time.
- Forces the model to commit to a structure, which measurably reduces hallucination
  compared to open-ended prose for the same extraction task.
- Gets you typed fields (numbers, dates) for free, and sortable/orderable output.
- Makes relationships between fields explicit rather than implicit in prose.

**Known cost/tradeoff:** structured formats like JSON are more token-expensive than
plain text for the same information, which raises cost/latency and — more
importantly — raises the risk that truncation (hitting the token limit mid-generation)
produces invalid, unparseable JSON (missing closing braces/brackets). If you're
generating structured output near your token budget, either raise the limit with
headroom, or use a JSON-repair step (e.g. the `json-repair` library) to salvage
truncated output rather than discarding it outright.

**Structuring input too, not just output:** the same idea applies in reverse — when
you have structured data to give the model (e.g. a product catalog entry), provide it
against an explicit schema (JSON Schema or similar) rather than as loose prose. This
gives the model a clear blueprint of expected fields/types, focuses its attention on
what's relevant, and helps it use fields like dates/timestamps correctly. Useful
specifically when integrating LLMs into a pipeline with well-defined data rather than
freeform user text.

## Document every prompt attempt
Prompt outputs vary across models, model versions, and even repeated calls to the
same model at the same settings (tied-probability tokens can break differently).
Without a record, you can't reliably tell whether a later change helped, hurt, or
was noise. Track, per attempt: name/version, goal (one sentence), model + version,
temperature/top-K/top-P/token limit, the full prompt text, and the output(s)
observed. For RAG-backed prompts, also capture the retrieval query, chunking
settings, and what chunks were actually inserted — the retrieval pipeline is a
moving part just as much as the prompt text.

Practical setup: keep this in a shared, structured place (a spreadsheet, or your
prompt-management tool of choice) rather than in scratch notes, so a team can compare
across attempts and re-run past prompts with one click rather than reconstructing
them from memory. Once a prompt is close to final, store it in its own file in the
codebase — separate from application code — so it's independently versionable and
testable, and set up automated evaluation against a test set rather than relying on
spot-checks.

## Get multiple people to try the same prompt
If you have more than one person available for a hard prompting problem, have
several independently attempt it following the same best practices. Even with
identical guidance, different attempts show real variance in quality — worth
comparing rather than committing to the first reasonable-looking prompt.

## Practical takeaway
Most of these are "cheap to apply, easy to skip" — simplicity, specificity,
positive framing, and documentation cost little extra effort per prompt but compound
over an iterative process. Treat prompt engineering explicitly as an iterative loop
(write → test → document → refine) rather than a one-shot task, and re-open that
loop whenever the model or model version changes.
