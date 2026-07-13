# Tree of Thoughts (ToT)

> **Notebook:** [`notebooks/patterns/01_tree_of_thoughts.ipynb`](../../notebooks/patterns/01_tree_of_thoughts.ipynb) · **Example:** [`examples/patterns/01_tree_of_thoughts.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/01_tree_of_thoughts.py)

Tree of Thoughts (`prismal.agents.patterns.tree_of_thoughts`, SPEC-PAT-001) turns single-shot LLM reasoning into a search problem. Instead of committing to one chain of reasoning, the pattern builds a tree of `Thought` nodes: at each node a user-supplied `generate_fn` proposes N candidate next steps, an `evaluate_fn` scores each in `[0, 1]`, and a search strategy (`beam`, `bfs`, or `dfs`) decides which branches survive. Search stops early as soon as any thought reaches the quality `threshold`, so good answers do not pay for the full tree.

The notebook applies ToT to a fixed subset of **GSM8K** (grade-school math word problems, embedded so there is no network dependency). Multi-step arithmetic is exactly the terrain where a single greedy chain of thought goes wrong on step 2 and never recovers — exploring and pruning several solution branches is what ToT was designed for.

## What it demonstrates

- Driving `tree_of_thoughts()` with two injected callables: an LLM-backed `generate_fn` and a deterministic, answer-checking `evaluate_fn` — the framework never dictates how thoughts are produced or judged.
- The factory-injection pattern used across prismal: business logic accepts callables, so the same search code runs against a real LLM or a test stub.
- Early termination via `threshold`: the search returns as soon as a thought scores >= 0.9 instead of exhausting the depth budget.
- Reading a `ToTResult`: the winning `Thought`, the root-to-best `best_path`, and `total_thoughts_generated` as a cost proxy.
- A head-to-head comparison of the `beam`, `bfs`, and `dfs` strategies on the same problem.

## How it works

### The two injected callables

Everything the pattern needs from you is a generator and an evaluator. The generator asks the LLM for three distinct solution approaches per node, using the path explored so far as refinement context:

```python
async def generate_solution_steps(problem: str, state: dict, path_so_far: list) -> list[str]:
    settings = get_settings()
    llm = ProviderRegistry(settings=settings).get_llm()

    context = ""
    if path_so_far and len(path_so_far) > 1:
        prev = "\n".join(f"  Step {i}: {t.content[:200]}" for i, t in enumerate(path_so_far[1:], 1))
        context = f"\n\nPreviously explored steps:\n{prev}"

    system_prompt = (
        "You are an expert in mathematics. Given a problem, propose 3 "
        "different and concise approaches to solve it step by step. "
        "Return exactly 3 approaches separated by '|||'. ..."
    )
    response = await llm.ainvoke([...])
    candidates = [c.strip() for c in str(response.content).split("|||") if c.strip()]
    return candidates[:3]
```

The signature matches `GenerateThoughtsFn`: `(problem, state, path_so_far) -> list[str]`. `state` is opaque to the framework — here it carries the original question and the expected answer through to both callables.

The evaluator is where GSM8K pays off: because each sample ships with a ground-truth answer, scoring can be cheap and deterministic instead of another LLM call:

```python
async def evaluate_solution(thought_text: str, state: dict) -> float:
    score = 0.0
    expected = state.get("answer", "").strip()

    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", thought_text)
    if numbers:
        score += 0.4          # a clearly identifiable final number
        if any(n.replace(",", "").replace(".", "") ==
               expected.replace(",", "").replace(".", "") for n in numbers):
            score += 0.5      # the number matches the expected answer

    if len(thought_text) > 50:
        score += 0.1          # explicit reasoning, not just a bare number
    return min(1.0, score)
```

A thought that reaches the correct answer with visible reasoning scores 1.0 — comfortably above the 0.9 threshold — which is what triggers early termination.

### Running the search

`solve_with_tot()` wires the callables into `tree_of_thoughts()` with a small, bounded search budget:

```python
return await tree_of_thoughts(
    problem=sample["question"],
    generate_fn=generate_solution_steps,
    evaluate_fn=evaluate_solution,
    state=state,
    breadth=3,       # 3 candidates per node (hint to the generator)
    depth=3,         # max tree depth
    beam_size=2,     # keep top-2 per level in beam mode
    threshold=0.9,   # stop as soon as any thought scores >= 0.9
    search_strategy=strategy,   # "beam" | "bfs" | "dfs"
)
```

Note that `breadth` is a *hint* the generator honours (it returns 3 candidates); the framework does not truncate. `depth` and `beam_size` are hard bounds — beam search costs at most `beam_size * depth` generate calls, which is what makes it the default.

### Inspecting the result

The returned `ToTResult` exposes the whole search, not just the answer. The notebook prints the winning thought, then walks `best_path` from the root down:

```python
result = await solve_with_tot(sample, strategy="beam")
print(f"Best thought (score={result.best_thought.score:.2f})")
print(f"Total thoughts generated: {result.total_thoughts_generated}")
for step in result.best_path:
    prefix = "ROOT" if step.depth == 0 else f"D{step.depth}"
    print(f"  {prefix} [score={step.score:.2f}] {step.content[:100]}...")
```

Each `Thought` carries `content`, `score`, `depth`, `parent_id`, and `children`, so the tree is fully reconstructable — useful when you want to audit *why* a branch won.

### Comparing strategies

The final cell reruns problem 1 under all three strategies and compares score against `total_thoughts_generated`:

```python
for strat in ["beam", "dfs", "bfs"]:
    r = await solve_with_tot(GSM8K_SAMPLES[0], strategy=strat)
    print(f"{strat:5s}: score={r.best_thought.score:.2f}  thoughts={r.total_thoughts_generated}")
```

The trade-off to look for: `bfs` expands every node at every level (exponential in breadth — thorough but expensive), `dfs` descends the highest-scoring branch first and pairs well with the early-stop threshold, and `beam` sits in between with a bounded per-level frontier. On easy GSM8K problems all three typically hit the threshold at depth 1, so the thought counts converge — the differences show up on harder, deeper searches.

## Key API

| Symbol | Role |
|---|---|
| `tree_of_thoughts()` | Async entry point: explores the reasoning tree, returns a `ToTResult`; raises `ValueError` on bad bounds and `ToTError` if the search yields no thoughts. |
| `ToTResult` | Dataclass: `best_thought`, `best_path` (root → best), `all_thoughts`, `total_thoughts_generated`. |
| `Thought` | One tree node: `content`, `score`, `depth`, `parent_id`, `children`, `is_terminal`, auto-generated `id`. |
| `GenerateThoughtsFn` | Callable contract `(problem, state, path_so_far) -> list[str]` for candidate generation. |
| `EvaluateThoughtFn` | Callable contract `(content, state) -> float in [0, 1]` for scoring. |
| `ProviderRegistry` | Resolves the configured LLM (`.get_llm()`) inside `generate_fn` — the only place provider access happens. |
| `get_settings()` | Loads prismal settings to construct the `ProviderRegistry`. |

Also worth knowing (not used in the notebook): `make_tot_node()` wraps the same search as a LangGraph-compatible node, and `budget_guard_fn` lets Budget Governance stop the search from deepening mid-run.

## Run it

```bash
uv run jupyter lab notebooks/patterns/01_tree_of_thoughts.ipynb
# or the script version:
uv run python examples/patterns/01_tree_of_thoughts.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — the generator calls a real LLM.

## Related

- [LATS (MCTS over actions)](03_lats.md) — the other tree-search pattern: MCTS with UCB1 over agent *actions* instead of reasoning steps.
- [Debate / Society of Mind](02_debate.md) — improves answers via multi-agent disagreement rather than tree exploration.
- [Budget Governance](../budget_governance.md) — `tree_of_thoughts()` honours a `budget_guard_fn` so cost caps can prune the search.
