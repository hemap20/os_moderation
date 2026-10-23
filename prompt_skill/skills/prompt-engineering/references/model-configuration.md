# LLM Output Configuration (Sampling Parameters)

Distilled from Google's "Prompt Engineering" whitepaper (Feb 2025). Prompt wording is
half the story — output configuration is the other half, and the two interact.

## Output length
- Setting a token limit doesn't make the model more concise in *style* — it just
  truncates once the limit is hit. If you need genuinely shorter output, ask for
  brevity in the prompt itself; don't rely on the token cap alone, or you risk
  cutting a response off mid-thought.
- Watch for this especially in agentic/ReAct-style prompting, where a model can keep
  emitting tokens (e.g. rambling after it already reached the useful answer) unless
  the output length is bounded.
- More generated tokens = more compute = higher latency and cost. Size the limit to
  the task, not to "just in case."

## Temperature, top-K, top-P
These three settings jointly control how randomly the next token is chosen from the
model's predicted probability distribution.

- **Temperature**: controls randomness. `0` = greedy/deterministic (always pick the
  highest-probability token — though ties can still break arbitrarily). Higher
  temperature flattens the distribution, making less-likely tokens more competitive.
- **Top-K**: only consider the K most probable next tokens. K=1 is equivalent to
  greedy decoding. Higher K = more variety/creativity; lower K = more
  conservative/factual.
- **Top-P (nucleus sampling)**: only consider tokens whose cumulative probability
  mass reaches P. P=0 ≈ greedy; P=1 = consider everything.

**How they combine:** when multiple are available, tokens must pass both the top-K
*and* top-P filters to be candidates, and temperature is then applied to sample among
survivors. At an extreme setting, one parameter can make the others irrelevant (e.g.
temperature=0 always outputs the top token regardless of top-K/top-P; top-K=1 always
outputs that one token regardless of temperature/top-P).

**Practical starting points:**
- Balanced/general use: temperature 0.2, top-P 0.95, top-K 30.
- More creative output: temperature 0.9, top-P 0.99, top-K 40.
- More conservative/factual output: temperature 0.1, top-P 0.9, top-K 20.
- Tasks with one objectively correct answer (math, classification, extraction,
  most reasoning/CoT tasks): temperature 0 — see CoT best practices in
  `reasoning-techniques.md` for why this matters specifically for reasoning tasks.

**Known failure mode — repetition loops:** a model can get stuck repeating the same
filler word/phrase/structure until the output window fills. This happens at *both*
extremes: at very low temperature the model can rigidly loop back into text it
already generated; at very high temperature the space of "acceptable" next tokens is
so large that a random choice can loop back into a prior state by chance. If you see
this, it's a signal to retune temperature/top-K/top-P rather than a prompt-wording
problem — sometimes the fix is content-side (better prompt), but the underlying
mechanism is sampling-side.

## Practical takeaway
Configuration and prompt wording are not independent — the same prompt at
temperature 0 vs. 1 can produce meaningfully different failure modes. When debugging
a bad output, check sampling config before assuming the prompt itself is at fault,
especially for tasks that should have a single correct answer (set temperature to 0
there) versus tasks that benefit from variety (raise temperature/top-K deliberately).
