# Reflection Loop

> **Notebook:** [`notebooks/patterns/06_reflection_loop.ipynb`](../../notebooks/patterns/06_reflection_loop.ipynb) · **Example:** [`examples/patterns/06_reflection_loop.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/06_reflection_loop.py)

The Reflection pattern (SPEC-035) is the simplest self-improvement loop: *generate* a draft, *critique* it against explicit criteria to get `(feedback, score)`, and if the score falls short of a threshold, *refine* — feed the previous draft and the critique back into the generator and try again, up to `max_iterations`. The loop always returns the highest-scoring draft it observed, so quality is monotone even when the threshold is never reached.

The notebook exercises the pattern on **technical-writing prompts** (in the spirit of the Reddit r/WritingPrompts dataset): explaining Transformers to Python developers, async/await for JavaScript programmers, RAG for product managers. Writing is the ideal domain for reflection because quality is *gradually improvable* against objective criteria — clarity, accuracy, completeness, length limits — which is exactly what the critique step scores.

## What it demonstrates

- The `reflection_loop()` coroutine driving generate → critique → refine cycles with a configurable quality `threshold` and `max_iterations` cap.
- Writing a real LLM-backed `generate_fn` that behaves differently on first draft versus refinement (when it receives `previous_draft` and `critique`).
- Writing a `critique_fn` that scores a draft against per-prompt evaluation criteria and returns `(feedback, score ∈ [0, 1])`.
- Threading task context through `AgentState.metadata` — the loop itself treats state as read-only.
- The `@with_reflection` decorator, which retrofits reflection onto any LangGraph subgraph node without restructuring its generation logic.

## How it works

### The generator: draft or refine

`generate_fn` must accept the state as its first argument and *may* accept `previous_draft` and `critique` keyword arguments. The notebook's generator branches on their presence — a fresh draft on iteration 1, a rewrite that incorporates the feedback on later iterations:

```python
async def generate_article(state, previous_draft=None, critique=None) -> str:
    llm = ProviderRegistry(settings=get_settings()).get_llm()
    instruction = state.get("metadata", {}).get("instruction", "Write about AI")

    if previous_draft and critique:
        system = ("You are an expert technical writer. Your previous draft was "
                  "evaluated... Rewrite the article incorporating the suggested "
                  "improvements. Keep what already works well.")
        user = (f"Original instruction: {instruction}\n\n"
                f"Your previous draft:\n{previous_draft}\n\n"
                f"Feedback received:\n{critique}\n\n...")
    else:
        system = "You are an expert technical writer..."
        user = f"Instruction: {instruction}"

    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    return str(response.content).strip()
```

### The critic: criteria in, (feedback, score) out

`critique_fn(draft, state)` returns a tuple: actionable textual feedback plus a float score in `[0, 1]`. The notebook's critic reads the prompt-specific criteria from `state["metadata"]["criteria"]` — e.g. for the RAG-for-PMs prompt: *no excessive jargon, RAG vs fine-tuning comparison, concrete use cases, ≤ 300 words* — and instructs the LLM to put the score alone on the first line:

```python
system = (
    "You are an expert technical editor. Evaluate the text against the given "
    "criteria. Provide:\n"
    "1. A score from 0.0 to 1.0 (on the FIRST line, only the number).\n"
    "2. Specific and actionable feedback on what to improve.\n"
    "Be strict: a score of 0.9+ only if the text is excellent."
)
...
try:
    score = max(0.0, min(1.0, float(lines[0].strip())))
except ValueError:
    numbers = re.findall(r"\b0\.\d+\b|\b1\.0\b", raw)   # fallback: first float in the text
    score = float(numbers[0]) if numbers else 0.5
```

The defensive parsing (clamping plus a regex fallback) matters because the loop's control flow hinges on this number. Keeping criteria explicit is equally important for convergence — vague critiques produce vague rewrites.

### Running the loop

The driver seeds an `AgentState` with the instruction and criteria, then hands both callables to `reflection_loop()`:

```python
state: AgentState = create_initial_state(session_id="nb-reflection-loop")
state["metadata"]["instruction"] = prompt["instruction"]
state["metadata"]["criteria"] = prompt["evaluation_criteria"]

best_draft, final_score = await reflection_loop(
    generate_fn=generate_article,
    critique_fn=critique_article,
    state=state,
    threshold=threshold,      # e.g. 0.80–0.88
    max_iterations=3,
)
```

The loop's contract: score ≥ threshold returns immediately; otherwise it refines until `max_iterations`, then returns the *best* draft observed (not necessarily the last). The notebook runs the same loop with three thresholds (0.80, 0.88, 0.75) to show how strictness controls the iteration count — a permissive threshold accepts the first draft, a strict one drives two or three refinements.

Two operational safeguards live in the implementation: the global setting `reflection_enabled=False` bypasses the loop entirely (first draft returned with sentinel score `1.0` — handy for latency-sensitive production), and the caller's `max_iterations` is capped by `get_settings().reflection_max_iterations`, so operators keep the ceiling. An optional `budget_guard_fn` is consulted before each *refinement* (iteration 1 always runs), letting a budget cap stop the loop with the best draft so far.

### Retrofitting nodes with `@with_reflection`

Any async LangGraph node that returns a string draft can opt into reflection with a decorator instead of calling the loop manually:

```python
from prismal.agents.patterns.reflection import with_reflection

@with_reflection(threshold=0.85, max_iterations=2, critique_fn=my_critic)
async def writer_node(state: AgentState) -> str:
    return await llm.generate(state)
```

The decorator wraps the node in `reflection_loop()` and stores `reflection_score` under `state["metadata"][<node_name>]` so downstream nodes and tests can inspect the outcome. Note that `critique_fn` is *required* at decoration time — there is deliberately no default critic, because evaluation logic is always domain-specific.

## Key API

| Symbol | Role |
|---|---|
| `reflection_loop()` | The coroutine: `(generate_fn, critique_fn, state, threshold=0.85, max_iterations=3, budget_guard_fn=None) -> (best_draft, final_score)`. |
| `with_reflection()` | Decorator factory retrofitting reflection onto async LangGraph node functions; requires an explicit `critique_fn`. |
| `GenerateFn` / `CritiqueFn` | Type aliases: `generate_fn(state, previous_draft=None, critique=None) -> str` and `critique_fn(draft, state) -> (feedback, score)`. |
| `AgentState` / `create_initial_state()` | The LangGraph state carrying the instruction and criteria under `metadata`; the loop treats it as read-only. |
| `ProviderRegistry` / `get_settings` | Resolve the LLM used by the notebook's generator and critic callables. |

## Run it

```bash
uv run jupyter lab notebooks/patterns/06_reflection_loop.ipynb
# or the script version:
uv run python examples/patterns/06_reflection_loop.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — both the generator and the critic make real LLM calls each iteration.

## Related

- [Constitutional AI](07_constitutional_ai.md) — reflection's principled sibling: self-revision driven by explicit principles plus an audit trail, instead of a free-form critic score.
- [Tree of Thoughts](01_tree_of_thoughts.md) — explores *multiple* candidate branches with evaluation, versus reflection's single draft refined in place.
- [Debate](02_debate.md) — externalizes the critique: other agents challenge the answer instead of a self-critique step.
- [Budget governance](../budget_governance.md) — how `budget_guard_fn` bounds refinement iterations under a token/cost budget.
