# Prompt Content & Prompt Assembly

Distilled from a practitioner-oriented book excerpt covering (1) what to put in a
prompt (content) and (2) how to structure/format it once gathered (assembly).

## 1. Static vs. dynamic content

Every prompt is made of two kinds of material:

- **Static content**: the same every time — task framing, clarifying instructions,
  persona/preamble, few-shot examples. This is what defines *the general problem* the
  model is meant to solve.
- **Dynamic content**: different per-request — the actual user input, retrieved facts,
  conversation history, anything specific to *this* invocation.

The line isn't always crisp (an instruction can double as context depending on how
the app is built), but the framing is useful for auditing a prompt: check that you
have enough of the static half to define the task unambiguously, and enough of the
dynamic half to give the model what it needs about the specific case.

### Clarifying the question (static)
Vague task framing is a common root cause of bad outputs. A one-line clarifying
addendum ("...for fun, not a textbook") does more to fix bad outputs than almost
any other prompt change, because it removes an entire category of misinterpretation
the model would otherwise have to guess at.

### Preamble / instructions (static)
A system-message-style block establishing role, tone, boundaries, and behavioral
rules (the analogy used is a leaked persona-instruction document for an assistant
persona). Practical notes:
- Put explicit behavioral instructions in the system message/preamble when the API
  supports role separation — models attend to it differently than to content mixed
  into the user turn.
- Not all models follow instructions equally reliably — test the actual instructions
  you plan to use against the actual model, don't assume portability across models.

### Few-shot prompting (static, but example-dependent)
- Few-shot examples primarily teach *format and style*, not just task boundaries —
  if you want a specific output shape, showing beats describing.
- Examples should be representative of the actual distribution of real inputs you
  expect, not hand-picked "nice" cases — a skewed example set biases the model's
  outputs toward the pattern in the examples, an effect similar to the cognitive
  anchoring bias in humans (e.g., an unrepresentative age-estimation example set will
  visibly shift the model's estimates on unrelated inputs).
- Small numbers of examples risk the model extrapolating a spurious pattern that only
  looks meaningful by chance — cover the major classes/variations in the task, not
  just a couple of typical instances.
- Trade-off: few-shot prompting scales poorly as context grows (cost, latency, and
  the "lost in the middle" effect — see below) and can bias output style/content
  toward the examples shown. Use it when you have genuinely representative examples
  illustrating a specific aspect of the task; don't reach for it as a default.

## 2. Dynamic content: sourcing and retrieval

Dynamic content needs a *pipeline* to gather at request time. Key considerations
when designing that pipeline:

- **How fast do you need it?** Content sources vary by urgency (e.g. a batch email
  digest tolerates slow retrieval; live autocomplete does not). Design retrieval
  latency budgets around the actual interactivity requirement, not a generic
  "as fast as possible."
- **How stable is the source?** More volatile sources (real-time data) are harder to
  pre-compute/cache; more stable sources (user profile, static docs) can be prepared
  ahead of time. Sort your context sources by volatility and treat each differently
  (precompute the stable ones, fetch-on-demand the volatile ones).
- **Mind-mapping as a design tool**: before building retrieval, sketch the tree of
  questions a human expert would ask to solve this task, including likely follow-ups —
  this surfaces what dynamic content categories you actually need to source, rather
  than guessing.

### Retrieval-Augmented Generation (RAG)
Necessary because LLMs can't natively access content outside training data /
context window.

- **Lexical retrieval** (keyword/term overlap, e.g. Jaccard similarity over
  stemmed/stopword-filtered text): cheap, transparent, tunable, but blind to
  synonyms/paraphrase.
- **Neural (embedding-based) retrieval**: converts both query and documents into
  vectors, retrieves by vector similarity — captures semantic similarity lexical
  methods miss, but the representation is opaque and harder to tune/debug.
- Practical takeaway: neither approach is strictly superior — lexical retrieval is
  easier to reason about and tune to specific user needs; neural retrieval covers
  paraphrase/synonym cases lexical can't. Many production RAG systems benefit from
  combining both rather than picking one.
- **Snippetizing**: break source documents into retrievable chunks. Chunking at
  natural boundaries (sections, paragraphs) generally beats arbitrary fixed-length
  windows, and augmenting a snippet with surrounding context (e.g. document title,
  section heading) improves retrieval usefulness beyond the raw snippet text alone.
- Treat your search query itself like a miniature prompt — the same clarity
  principles that improve a full prompt also improve a retrieval query.

### Summarization (for content too long to include directly)
- **Hierarchical summarization**: summarize chunks, then summarize the summaries,
  recursively, for content that doesn't fit context in one pass. Watch for
  information loss/drift compounding at each level — verify the final summary still
  reflects source intent, especially for content with many equally-important
  subtopics (a known failure mode: broad topics can get flattened unevenly across
  levels).
- **General vs. specific summaries**: a summary written toward a specific downstream
  question is more useful than a generic summary, when you know the question in
  advance — bias the summarization prompt toward what will actually be asked, rather
  than reaching for a general-purpose summary every time.

## 3. Prompt assembly / structure

Once content is gathered, structure matters for how reliably the model uses it.

### Position effects ("lost in the middle")
- Content near the *start* and *end* of a prompt gets attended to more reliably than
  content buried in the middle — the "lost in the middle" / "Valley of Meh" effect.
  This gets worse as prompts get longer.
- Practical fix: put your most important instruction/context near the start and
  *repeat a short reminder of the core task/instruction near the end*, right before
  the completion point — don't rely on something stated once in the middle of a long
  prompt to still be salient at generation time.

### The "sandwich" structure
A generically useful layout: **Introduction** (frame the task) → **Context**
(the bulk dynamic content) → **Transition/refocus** (a short reminder of what's being
asked, positioned right after the context dump) → **Prompt** (the actual final ask).
The refocus step exists specifically to counteract the lost-in-the-middle effect
after a large context block.

### Conversation framing ("Advice Conversation")
Framing the prompt as a dialogue (one party asking for help, one party — the model —
providing it) works for both chat-native models and completion models (via manual
transcript formatting), and extends naturally to multi-round interactions where you
inject app logic between turns, and to tool-use/agentic loops.

### Choosing a document format
Several formats trade off differently — pick based on task needs, not habit:
- **Freeform text**: flexible, good for inserting arbitrary information, but weaker
  for long or highly structured elements.
- **Transcript/script format** (labeled turns, e.g. "User: ... / Assistant: ..."):
  easy to assemble, natural for conversation, less ideal for structured data.
- **Structured/markerless formats** (explicit field markers, e.g. templated
  Q/A blocks): better for tasks needing precise, parseable structure.
- **Markdown**: a strong default for structured-but-natural-language content —
  models have seen enormous amounts of Markdown in training, headings communicate
  hierarchy, indentation is forgiving, and links/tables are natively understood.
  A table of contents at the top of a long Markdown document helps orient the model
  the same way it helps a human reader.
- **Structured markup (YAML, JSON, etc.)**: use when you need strict machine-parseable
  structure or are working with tools/artifacts that require a formal schema — but
  it costs more tokens and is less natural for prose-heavy content than Markdown.

### Snippet design principles (for reusable content chunks)
When building reusable pieces of context (e.g. retrieved snippets, few-shot blocks),
aim for:
- **Modularity** — a snippet should be insertable/removable independently without
  breaking the surrounding prompt.
- **Naturalness** — it should read as an organic part of the document, not an
  obviously bolted-on fragment.
- **Brevity** — don't pad; every token costs context budget and dilutes attention.
- **Inertness** — a snippet's token cost should be computable once and treated as
  fixed cost, not something that needs re-measurement per use (relevant when working
  close to context-window limits).

### Relationships among prompt elements
When assembling multiple pieces of content into one prompt, track:
- **Position** — where each element goes (interacts directly with the lost-in-the-
  middle effect above).
- **Importance** — not all content is equally essential; explicitly ranking elements
  by importance lets you decide what to cut first under context-length pressure.
- **Dependency** — some elements require others to make sense (requirements), and
  some are mutually exclusive (incompatibilities) — track these explicitly if you're
  dynamically assembling prompts from a pool of optional elements, so you don't end
  up including contradictory or dangling content.

## Practical takeaway for prompt engineering work
Treat prompt-building as two separate passes: (1) gather content — deliberately
distinguish static task-definition material from dynamic per-request material, and
build a retrieval/summarization pipeline sized to actual latency and volatility needs;
(2) assemble — choose a document format matched to the task, structure long prompts
with a sandwich pattern to fight the lost-in-the-middle effect, and explicitly manage
position/importance/dependency when a prompt is assembled from multiple optional
pieces rather than written by hand each time.
