# Reasoning-Elicitation Techniques

Distilled from: Wei et al. 2022 "Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models" (CoT); Yao et al. 2023 "ReAct: Synergizing Reasoning and
Acting in Language Models" (ReAct); Yao et al. 2023 "Tree of Thoughts: Deliberate
Problem Solving with Large Language Models" (ToT).

## Chain-of-Thought (CoT) Prompting

**What it is:** Instead of few-shot exemplars that go straight from question to answer,
give exemplars that show the intermediate reasoning steps leading to the answer. The
model then generates its own reasoning steps before answering on new inputs.

**When it helps most (all three conditions together):**
1. The task requires multi-step reasoning (not a single lookup or one-step calc).
2. You're using a large/capable model — CoT provides little or no benefit on small
   models, and can even hurt performance on them. This is not a universal "add
   reasoning steps" trick; it's conditional on model capability.
3. Standard prompting has a flat performance curve on the task (i.e., there's real
   headroom to gain).

If a task is easy enough that standard prompting already does well, CoT adds little.

**Practical prompt-construction notes:**
- Robustness: CoT gains hold up across different annotators' writing styles, different
  exemplar sets, and different exemplar orderings — the *presence* of step-by-step
  reasoning matters more than its exact phrasing.
- But annotation quality still matters at the margin — a poorly-written chain of
  thought underperforms a well-written one. Don't treat "add any reasoning" as
  license to be sloppy.
- Natural-language reasoning outperforms equivalent shortcuts like "generate the
  equation only" or "generate N placeholder tokens matching compute cost" — the
  benefit isn't just from added compute/tokens, it's from language-mediated
  intermediate structure.
- Putting the reasoning *before* the answer matters — reasoning generated after the
  answer doesn't help, confirming the model actually uses the chain to reach the
  answer rather than the chain just being decorative.
- Failure modes to watch for: semantic misunderstanding, missing a step, and
  (for arithmetic) plain calculator errors — pairing CoT with an external calculator
  for arithmetic sub-steps measurably improves accuracy.
- On multiple-choice/binary tasks, a correct final answer doesn't guarantee correct
  reasoning (the model can luck into the right choice) — don't treat surface accuracy
  as proof the reasoning path was sound if you need explainability.

**Practical takeaway for prompt engineering work:** when a task is complex and using a
capable model, explicitly ask for step-by-step reasoning before the final answer (or
supply few-shot exemplars that do so). For simple/lookup-style tasks, skip it — it
adds latency/cost with no accuracy benefit and occasionally hurts.

**Note on temperature:** for single-pass CoT, set temperature to 0 — reasoning tasks
generally have one correct answer, and greedy decoding is the right match for that.
If single-pass CoT proves unreliable on an ambiguous/adversarial task, see
`core-prompting-techniques.md` for **self-consistency** (sampling many CoT paths at
high temperature and majority-voting the answer) as the next escalation.

## ReAct: Reasoning + Acting

**What it is:** Interleave reasoning ("thoughts") with concrete actions (e.g., tool
calls like search/lookup) and their resulting observations, in a repeating
Thought → Action → Observation loop, instead of doing pure reasoning (CoT) or pure
action generation with no reasoning in between.

**Why interleaving beats either alone:**
- Pure CoT reasoning is a closed box: it can't fetch new information, so it's prone to
  hallucinating facts it doesn't actually know, and errors compound with no way to
  correct course mid-stream.
- Pure action-generation (act-only, no reasoning) is prone to losing track of task
  state — it can fail to notice a goal is complete, forget what's already been tried,
  or loop on the same failed action, because nothing is explicitly tracking progress.
- ReAct's thoughts serve several concrete functions worth naming in a prompt/exemplar
  design: decomposing a goal into an ordered plan, tracking which subgoal is done and
  which is next, pulling out the relevant fact from a noisy observation, and deciding
  to reformulate/retry an action after a bad result.

**Design guidance for building a ReAct-style agent prompt:**
- Exemplars are full human-written trajectories: alternating Thought / Action /
  Observation steps, ending in a final answer action. No special format is required
  beyond "thought in plain language, then an action in a fixed action syntax."
- Thoughts don't need to appear at every single step for action-heavy tasks (e.g.
  navigating many small steps) — sparse thoughts placed at decision points (goal
  decomposition, subgoal transitions, exception handling) work better than a thought
  before every single micro-action, which can add noise without adding value.
- For knowledge-lookup-style tasks (Q&A, fact verification), denser thought-action
  interleaving (a thought before most actions) works better, since each step usually
  changes what's known.
- Known failure mode: the model can get stuck in a reasoning loop, repeating a
  previous thought/action rather than recognizing the approach has stalled — worth
  a guardrail (max steps, explicit "if repeating, try a different action" instruction)
  when using this pattern operationally.
- Combining ReAct with self-consistency / majority-voting (run ReAct, fall back to
  pure CoT-style sampling when it fails to produce an answer in budget, or vice versa)
  measurably beat either method alone in the original paper — worth considering a
  fallback path for production agents rather than a single fixed strategy.
- Human-in-the-loop editing: because thoughts are plain language, a human can directly
  edit a wrong or hallucinated thought mid-trajectory to steer the agent, which is far
  cheaper than re-writing a whole action sequence — a useful debugging/correction
  affordance to build into interactive agent tooling.

**Practical takeaway for prompt engineering work:** ReAct is the right pattern any time
you're building an agent/bot that needs external tools (search, APIs, retrieval, code
execution) *and* multi-step reasoning about what to do with results — e.g. a
conversational bot that needs to look things up mid-dialogue. Give it a
Thought/Action/Observation exemplar format, and decide thought density (sparse vs.
dense) based on how often new information changes what the model needs to reason about.

## Tree of Thoughts (ToT)

**What it is:** A generalization of CoT for problems where a single linear reasoning
chain isn't enough — problems that benefit from exploring *multiple* candidate next
steps, evaluating them, and backtracking when a path stalls, rather than committing to
one left-to-right reasoning path. Concretely: the model generates several candidate
"thoughts" (intermediate reasoning steps) at each stage, self-evaluates how promising
each one is, and a search algorithm (breadth-first or depth-first) decides which
branches to keep exploring, prune, or backtrack from.

**Why plain CoT falls short for some tasks:** standard left-to-right generation commits
to its first few tokens/steps and can't reconsider — if an early step is wrong (e.g. a
bad first move in a math puzzle), the whole chain is often unrecoverable. This matters
most for tasks requiring search, planning, or lookahead, where the "right" next step
isn't obvious from local context alone. In the paper's benchmark task (Game of 24),
plain CoT solved only 4% of problems while ToT solved 74% — an unusually large gap that
signals this is the right technique for a specific *class* of problem, not a strict
upgrade over CoT in general.

**The four design choices when building a ToT setup:**
1. **Thought decomposition** — decide what counts as one "thought" (a step small
   enough to generate multiple diverse candidates of, but large enough to meaningfully
   evaluate). This is task-specific: a few words, one line of an equation, or a whole
   paragraph plan, depending on the problem.
2. **Thought generation** — either sample several candidates independently (good when
   the space of good next-thoughts is rich/diverse, e.g. creative writing), or prompt
   the model once to propose several distinct next steps together (good when the
   space is constrained, e.g. a math step, to avoid generating near-duplicates).
3. **State evaluation** — have the model *itself* judge how promising each candidate
   state is, either by scoring each independently (e.g. "sure/maybe/impossible", or a
   1–10 value) or by voting across a set of candidates to pick the best one. This
   replaces a hand-coded or trained heuristic with the LM's own judgment — cheaper to
   set up than a trained value model, though not perfectly reliable.
4. **Search algorithm** — breadth-first search (keep the top-b candidates at each
   step; good for shallow trees with a small number of steps) or depth-first search
   with pruning/backtracking (good for deeper trees where you want to commit to the
   most promising path but back out when the evaluator flags it as a dead end).

**When ToT is worth the (substantial) extra cost:**
- Tasks requiring genuine search/planning/lookahead, where an early misstep derails
  the whole solution and can't be locally corrected — not tasks a capable model
  already solves well with plain CoT. Don't reach for it as a default; it costs
  meaningfully more (the paper reports roughly 5–100x the generated tokens of a single
  CoT pass for a given task, so the cost-benefit only pays off when CoT genuinely
  struggles).
- Concretely, symptoms that suggest ToT over CoT: a large gap between "best of many
  independent CoT samples" and a single CoT sample (implies the model *can* find good
  answers but greedy left-to-right decoding usually picks a bad path early); or a task
  that inherently involves trying several options and backing out of dead ends (word
  puzzles, constrained search, some planning tasks).
- Not worth it: tasks a model already does well with direct prompting or plain CoT —
  the extra exploration doesn't move the needle and just burns tokens.

**Practical takeaway for prompt engineering work:** treat ToT as a heavier, more
expensive escalation from CoT — reach for it specifically when a task has a
"one wrong early step ruins everything, but there are recognizably better and worse
next moves" shape (planning, constrained search, some creative-writing coherence
tasks), and when you have budget for meaningfully more inference cost per task. For
most everyday tasks, plain CoT (or ReAct if tools are needed) is the better
cost/benefit choice.
