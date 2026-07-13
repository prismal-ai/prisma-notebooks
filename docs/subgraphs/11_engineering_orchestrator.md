# Engineering Orchestrator

> **Notebook:** [`notebooks/subgraphs/11_engineering_orchestrator.ipynb`](../../notebooks/subgraphs/11_engineering_orchestrator.ipynb) · **Example:** [`examples/subgraphs/11_engineering_orchestrator.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/11_engineering_orchestrator.py)

The engineering orchestrator is the **hierarchical domain orchestrator** (SPEC-042, Phase 40) for everything that writes, plans, or manipulates code and files. An LLM-driven `engineering_supervisor` node — the subgraph's entry point — routes each request to one of five leaf agents: `coder` (ReAct tool-calling, surgical edits), `codeact` (generates and runs Python in a sandbox, multi-step iterative coding), `planner` (SDD spec decomposition), `file_manager` (filesystem operations), or `skill_manager` (install/activate skills). Every leaf returns to the supervisor, whose built-in loop breaker ends the turn once a leaf has responded.

When `PRISMAL_HIERARCHICAL_MODE=true`, the root supervisor delegates any "build / implement / refactor / plan" request to this subgraph instead of picking individual leaves — keeping the root's routing prompt small and the domain decision (e.g. `coder` vs `codeact` by task complexity) close to the agents that know the difference.

## What it demonstrates

- Domain-level LLM routing across five engineering leaves, with each leaf edging back to the supervisor.
- The conditional-edge router `_engineering_router`, which reads `state["next_agent"]` and rejects anything outside `ENGINEERING_AGENTS`.
- Grounding a routing demo in a real dataset: GitHub issues from `langchain-ai/langchain` reclassified into the five leaves.
- Measuring routing accuracy against expected leaves — no LLM, fully deterministic.
- Rebuilding the orchestrator topology as a runnable LangGraph with stub leaves and `MemorySaver` checkpointing.

## Pipeline topology

```
engineering_supervisor  ◄── entry point (domain LLM router)
      │ conditional edge: _engineering_router(state) → next_agent | __end__
  ┌───┼─────────┬───────────┬──────────────┐
  ▼   ▼         ▼           ▼              ▼
coder  codeact  planner  file_manager  skill_manager
  │     │         │          │             │
  └─────┴─────────┴──────────┴─────────────┘
      │ every leaf returns to the supervisor
      ▼
engineering_supervisor ──(loop breaker: leaf already answered)──► END
```

## How it works

### Dataset — GitHub issues reclassified into five leaves

The notebook first locates the repo's `data/` folder (walking up from the cwd), then loads open issues from `data/github-issues/github_issues.csv`. Real issue prose is ideal routing input: natural phrasing, no synthetic patterns. A heuristic assigns each issue the leaf it *should* land on, and an embedded six-task fallback kicks in if the CSV is missing:

```python
def _classify(title: str, body: str) -> str:
    """Demo heuristic to assign the expected leaf based on content."""
    text = f"{title} {body}".lower()
    if re.search(r"\b(spec|design|architecture|prd|plan)\b", text):
        return "planner"
    if re.search(r"\b(file|path|rename|move|directory|workspace)\b", text):
        return "file_manager"
    if re.search(r"\b(skill|install|activate|register|plugin)\b", text):
        return "skill_manager"
    if re.search(r"\b(traceback|run this|execute|script|notebook|stdout)\b", text):
        return "codeact"
    return "coder"
```

### Demo 1 — routing simulation, no LLM

`simulate_supervisor()` reuses the same heuristic as a stand-in for the LLM supervisor. Each task is routed, dispatched to a canned leaf stub (`stub_coder`, `stub_codeact`, `stub_planner`, `stub_file_manager`, `stub_skill_manager` — each returning a `LeafResult` with a summary and artifacts), and compared against the expected leaf:

```python
for t in tasks:
    routed = simulate_supervisor(t.request)
    ok = routed == t.expected_leaf
    hits += int(ok)
    result = LEAVES[routed](t)
    ...
print(f"Routing accuracy: {hits}/{len(tasks)} ({pct:.1f}%)")
```

### Demo 2 — real LangGraph with stubs and MemorySaver

The notebook imports the *real* router and agent tuple from the prismal builder and hand-assembles the identical topology with stub leaves — so the actual conditional edge, return edges, and loop-breaker semantics execute under LangGraph with checkpointing, without any API key:

```python
from prismal.agents.subgraphs.engineering_orchestrator.builder import (
    ENGINEERING_AGENTS,
    _engineering_router,
)

sg = StateGraph(DemoState)
sg.add_node("engineering_supervisor", demo_supervisor)
for leaf in ENGINEERING_AGENTS:
    sg.add_node(leaf, _make_leaf_node(leaf))
    sg.add_edge(leaf, "engineering_supervisor")
sg.set_entry_point("engineering_supervisor")
sg.add_conditional_edges("engineering_supervisor", _engineering_router)
compiled = sg.compile(checkpointer=MemorySaver())
```

The demo supervisor implements the loop breaker explicitly: if the last message is an `AIMessage` and `current_agent` is one of the leaves, it returns `next_agent=None`, which the router maps to `__end__`:

```python
async def demo_supervisor(state: DemoState) -> dict[str, Any]:
    msgs = list(state.get("messages") or [])
    if msgs and getattr(msgs[-1], "type", "") == "ai" \
            and state.get("current_agent") in ENGINEERING_AGENTS:
        return {"current_agent": "engineering_supervisor", "next_agent": None}
    human = next((m for m in reversed(msgs) if getattr(m, "type", "") == "human"), None)
    return {
        "current_agent": "engineering_supervisor",
        "next_agent": simulate_supervisor(human.content if human else ""),
    }
```

### The real builder

In the prismal source, `_make_definition()` wires the production nodes (`coder_node`, `codeact_node`, `planner_node`, `file_manager_node`, `skill_manager_node`) plus a supervisor produced by `make_domain_supervisor(domain_name="engineering", ...)` into a `SubgraphDefinition`; `SubgraphFactory.build()` then compiles it with an isolated SQLite checkpointer. `register_engineering_orchestrator()` installs it idempotently in the global `SubgraphRegistry`; `get_compiled_engineering_orchestrator()` returns a process-cached compiled graph. The topology deliberately mirrors the research orchestrator — same supervisor/router shape, different leaf set.

## Key API

| Symbol | Role |
|---|---|
| `get_compiled_engineering_orchestrator(...)` | Returns (building if needed) the compiled orchestrator; module-level cache |
| `register_engineering_orchestrator(...)` | Compiles and registers the orchestrator in `SubgraphRegistry` (idempotent) |
| `_make_definition()` | Builds the `SubgraphDefinition` (entry point `engineering_supervisor`, 5 leaves, return edges) |
| `make_domain_supervisor("engineering", ...)` | Produces the LLM router node with built-in loop breaker |
| `_engineering_router(state)` | Conditional edge: `state["next_agent"]` → leaf name or `__end__` (unknown values end cleanly) |
| `ENGINEERING_AGENTS` | `("coder", "codeact", "planner", "file_manager", "skill_manager")` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/11_engineering_orchestrator.ipynb
uv run python examples/subgraphs/11_engineering_orchestrator.py   # from the prismal repo
```

No API keys required — both demos run on deterministic stubs and mocks (the GitHub-issues CSV is local, with an embedded fallback).

## Related

- [Analysis Orchestrator](10_analysis_orchestrator.md) — the sibling domain orchestrator for analytical output.
- [Research Orchestrator](12_research_orchestrator.md) — the sibling domain orchestrator for information gathering.
- [Dev Pipeline](03_dev_pipeline.md) — the full PO → Architect → Developer → QA pipeline these leaves complement.
- [Code Review](04_code_review.md) — a linear engineering pipeline, contrasted with this router-based topology.
