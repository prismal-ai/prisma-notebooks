# Analysis Orchestrator

> **Notebook:** [`notebooks/subgraphs/10_analysis_orchestrator.ipynb`](../../notebooks/subgraphs/10_analysis_orchestrator.ipynb) · **Example:** [`examples/subgraphs/10_analysis_orchestrator.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/10_analysis_orchestrator.py)

The analysis orchestrator is a **hierarchical domain orchestrator** (SPEC-042, Phase 40): a mid-level supervisor that sits between prismal's root supervisor and four analytical leaf agents. It does not execute tasks itself — an LLM-driven `analysis_supervisor` node classifies each request and routes it to `data_analyst`, `ml_pipeline`, `dev_pipeline`, or `financial_analyst`; each leaf hands control back, and a built-in loop breaker sends the turn to `END` once a leaf has answered. Notably, three of the four leaves are themselves *pre-compiled subgraphs* — LangGraph accepts a `CompiledStateGraph` anywhere a node callable is expected.

Use it when `PRISMAL_HIERARCHICAL_MODE=true`: instead of one flat root supervisor juggling 12+ agents in a single routing prompt, the root only knows three domain orchestrators, and each domain supervisor picks among 3–5 well-separated leaves — shorter prompts, higher routing accuracy, and per-domain iteration caps.

## What it demonstrates

- Two-level hierarchical routing: root supervisor → domain orchestrator → leaf agent, versus a flat 12+-agent prompt.
- Embedding pre-compiled subgraphs (`dev_pipeline`, `ml_pipeline`, `financial_analyst`) as nodes inside another graph.
- The conditional-edge router `_analysis_router`, which reads `state["next_agent"]` and rejects anything outside `ANALYSIS_AGENTS`.
- The supervisor loop-breaker pattern: leaf responds → supervisor detects it → routes to `END` (iteration cap = 8 per domain).
- Building a runnable demo graph with `StateGraph` + `MemorySaver` and stub nodes, so the wiring runs without any LLM.

## Pipeline topology

```
root_supervisor  (only in full hierarchical mode)
      │
      ▼
analysis_supervisor  ◄── entry point (domain LLM router)
      │ conditional edge: _analysis_router(state) → next_agent | __end__
  ┌───┼─────────────┬──────────────────┐
  ▼   ▼             ▼                  ▼
data_analyst  ml_pipeline*  dev_pipeline*  financial_analyst*
  │   │             │                  │      (* = pre-compiled subgraph)
  └───┴─────────────┴──────────────────┘
      │ every leaf returns to the supervisor
      ▼
analysis_supervisor ──(loop breaker: leaf already answered)──► END
```

## How it works

### Dataset — six Business Intelligence tasks

The demo ships an embedded "Business Intelligence Center" dataset: six analytical tasks that cover all four leaves (sales analytics, churn-model training, a FastAPI endpoint, a NVDA/TSLA stock analysis, DAU/ARPU metrics, and a JWT refactor). Each task records the leaf it *should* land on, so routing accuracy can be scored:

```python
@dataclass
class AnalyticsTask:
    id: str
    request: str
    domain: str  # data_analyst | ml_pipeline | dev_pipeline | financial_analyst
    complexity: str  # LOW | MEDIUM | HIGH
    expected_output: str
    context: str
```

### Simulating the domain supervisor without an LLM

In production the `analysis_supervisor` node (built by `make_domain_supervisor(domain_name="analysis", ...)`) calls the LLM to pick a leaf. The notebook reproduces that decision deterministically with keyword scoring, so every demo runs offline:

```python
def simulate_domain_supervisor(request: str) -> str:
    lower = request.lower()
    scores = {"data_analyst": 0, "ml_pipeline": 0,
              "dev_pipeline": 0, "financial_analyst": 0}
    for keywords, agent in _ROUTING_RULES:
        for kw in keywords:
            if kw in lower:
                scores[agent] += 1
    best = max(scores, key=lambda a: scores[a])
    return best if scores[best] > 0 else "data_analyst"
```

`demo_simulation()` routes all six tasks through this classifier, dispatches each to a canned leaf stub (`AgentResult` with summary, artifacts, metrics), and prints a routing-accuracy table.

### Demo 2 — hierarchical vs flat routing

`demo_comparison()` is pure narrative: it contrasts flat mode (the root supervisor's prompt enumerates 12+ agents) with hierarchical mode (root knows 3 orchestrators; `analysis_supervisor` knows 4 leaves). The trade-off it spells out: one extra LLM call for the intermediate hop, in exchange for shorter routing prompts, better accuracy, and per-domain iteration counters that cap local loops at 8.

### Demo 3 — a real LangGraph with stubs and MemorySaver

The most instructive part: the notebook imports the *real* router and agent tuple from the builder, then assembles the same topology by hand with stub leaves and a keyword supervisor — no LLM, real LangGraph execution with checkpointing:

```python
from prismal.agents.subgraphs.analysis_orchestrator.builder import (
    ANALYSIS_AGENTS,
    _analysis_router,
)

builder = StateGraph(DemoState)
builder.add_node("analysis_supervisor", demo_analysis_supervisor)
builder.add_node("data_analyst", data_analyst_stub_node)
# ... ml_pipeline, dev_pipeline, financial_analyst stubs ...

builder.set_entry_point("analysis_supervisor")
builder.add_conditional_edges("analysis_supervisor", _analysis_router)
for leaf in ANALYSIS_AGENTS:
    builder.add_edge(leaf, "analysis_supervisor")

graph = builder.compile(checkpointer=MemorySaver())
```

The demo supervisor also reproduces the loop breaker: if the last message is an `AIMessage` from a leaf, it sets `next_agent=None`, which `_analysis_router` maps to `__end__`. It tracks its iteration count in an isolated metadata slot, `state["metadata"]["domain_analysis"]["iteration_count"]`.

### The real builder — async because the leaves are compiled graphs

In the prismal source, `_make_definition()` is `async`: it awaits `get_compiled_dev_pipeline()`, `get_compiled_ml_pipeline()`, and `get_compiled_financial_analyst()` when the caller does not inject them, then returns a `SubgraphDefinition` that `SubgraphFactory.build()` compiles with an isolated SQLite checkpointer. Tests (and this notebook's Demo 4 printout) inject lightweight stubs instead:

```python
# With stubs for testing:
graph = await get_compiled_analysis_orchestrator(
    dev_pipeline_graph=my_dev_stub,
    ml_pipeline_graph=my_ml_stub,
    financial_analyst_graph=my_fin_stub,
)

# Production (lazy-builds the 3 pipelines):
graph = await get_compiled_analysis_orchestrator()
```

`register_analysis_orchestrator()` is the idempotent variant that also installs the definition in the global `SubgraphRegistry`. Demo 4 (`demo_hierarchy()`) closes the loop by rendering the full 3-level hierarchy map (root → 3 orchestrators → leaves) and re-running the routing table over the dataset.

## Key API

| Symbol | Role |
|---|---|
| `get_compiled_analysis_orchestrator(...)` | Returns (building if needed) the compiled orchestrator; accepts stub graphs for the three pipeline leaves |
| `register_analysis_orchestrator(...)` | Compiles and registers the orchestrator in `SubgraphRegistry` (idempotent) |
| `_make_definition(...)` | Async — awaits the three pipeline factories, returns the `SubgraphDefinition` |
| `make_domain_supervisor("analysis", ...)` | Produces the LLM router node `analysis_supervisor` with built-in loop breaker |
| `_analysis_router(state)` | Conditional edge: `state["next_agent"]` → leaf name or `__end__` (unknown values end cleanly) |
| `ANALYSIS_AGENTS` | `("data_analyst", "dev_pipeline", "ml_pipeline", "financial_analyst")` |
| `metadata["domain_analysis"]["iteration_count"]` | Per-domain isolated iteration counter (cap = 8) |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/10_analysis_orchestrator.ipynb
uv run python examples/subgraphs/10_analysis_orchestrator.py   # from the prismal repo
```

No API keys required — all four demo modes run on deterministic stubs and mocks.

## Related

- [ML Pipeline](01_ml_pipeline.md) — the `ml_pipeline` leaf this orchestrator routes training tasks to.
- [Financial Analyst](02_financial_analyst.md) — the `financial_analyst` leaf for market analysis tasks.
- [Dev Pipeline](03_dev_pipeline.md) — the `dev_pipeline` leaf for software-development tasks.
- [Engineering Orchestrator](11_engineering_orchestrator.md) — the sibling domain orchestrator for code and file work.
