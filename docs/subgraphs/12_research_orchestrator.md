# Research Orchestrator

> **Notebook:** [`notebooks/subgraphs/12_research_orchestrator.ipynb`](../../notebooks/subgraphs/12_research_orchestrator.ipynb) · **Example:** [`examples/subgraphs/12_research_orchestrator.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/12_research_orchestrator.py)

The research orchestrator is the smallest of prismal's three **hierarchical domain orchestrators** (SPEC-042): a `research_supervisor` node — the subgraph's entry point, built by `make_domain_supervisor()` — decides between exactly two information-gathering channels. External questions (literature surveys, paper summaries, comparisons) go to `researcher` (web/literature search); factual questions over an already-curated corpus go to `rag_agent` (internal vector-store Q&A). Both leaves return to the supervisor, whose built-in loop breaker guarantees the subgraph terminates after at most two supervisor calls per invocation.

In hierarchical mode (`PRISMAL_HIERARCHICAL_MODE=true`), the root supervisor delegates any investigation-flavoured request to this subgraph instead of picking individual leaves. With only two options, it is the clearest illustration of the domain-orchestrator pattern: a thin LLM router, a conditional edge, return edges, and a loop breaker.

## What it demonstrates

- The minimal domain-orchestrator shape: one LLM router, two leaves, return edges, loop-break to `END`.
- The `researcher` vs `rag_agent` dichotomy — external search versus internal knowledge-base lookup.
- Grounding routing in two real datasets that embody that dichotomy: arXiv paper titles and MedQuAD medical questions.
- The conditional-edge router `_research_router`, which reads `state["next_agent"]` and defaults unknown values to `__end__`.
- Rebuilding the topology as a runnable LangGraph with stub leaves and `MemorySaver` — no LLM required.

## Pipeline topology

```
research_supervisor  ◄── entry point (domain LLM router)
      │ conditional edge: _research_router(state) → next_agent | __end__
  ┌───┴────────┐
  ▼            ▼
researcher   rag_agent
(web / lit   (internal vector
 search)      store Q&A)
  │            │
  └─────┬──────┘
        │ both leaves return to the supervisor
        ▼
research_supervisor ──(loop breaker: leaf already answered)──► END
```

## How it works

### Datasets — the perfect routing dichotomy

After locating the repo's `data/` folder, the notebook draws tasks from two local CSVs. arXiv paper titles (`data/arxiv/arxiv_papers.csv`) become "summarise the contributions of..." requests, expected to route to `researcher`; MedQuAD questions (`data/medquad/medquad.csv`) are definitional medical questions over a curated knowledge base, expected to route to `rag_agent`. An embedded four-task fallback covers the case where neither CSV exists.

### Classifying requests without an LLM

Two heuristics stand in for the LLM supervisor's decision, so every demo runs offline. Definitional openers signal a KB lookup; survey/comparison vocabulary signals external research:

```python
def _looks_like_rag_query(text: str) -> bool:
    """Heuristic: factual / definitional questions → rag_agent."""
    t = text.lower()
    return bool(
        re.search(r"^\s*(what is|what are|how does|how do|why does|when is|"
                  r"what does|how is|what kind of)\b", t)
        or " kb " in t
        or "knowledge base" in t
        or "internal docs" in t
    )

def _classify(text: str) -> str:
    if _looks_like_rag_query(text):
        return "rag_agent"
    if _looks_like_research_query(text):
        return "researcher"
    # Default: if uncertain, short question → rag, long → researcher.
    return "rag_agent" if len(text) < 80 else "researcher"
```

### Demo 1 — routing simulation

`demo_simulation()` routes each task through `_classify()`, dispatches it to a canned leaf stub (`stub_researcher` returns synthesized findings with arXiv citations; `stub_rag_agent` returns KB chunks with `kb:medquad#...` citations), and scores accuracy against the expected leaf.

### Demo 2 — real LangGraph with stubs and MemorySaver

The notebook imports the *real* router and agent tuple from the prismal builder and hand-assembles the identical topology with stub leaves — actual LangGraph execution with checkpointing, no API key:

```python
from prismal.agents.subgraphs.research_orchestrator.builder import (
    RESEARCH_AGENTS,
    _research_router,
)

sg = StateGraph(DemoState)
sg.add_node("research_supervisor", demo_supervisor)
for leaf in RESEARCH_AGENTS:
    sg.add_node(leaf, _make_leaf_node(leaf))
    sg.add_edge(leaf, "research_supervisor")
sg.set_entry_point("research_supervisor")
sg.add_conditional_edges("research_supervisor", _research_router)
compiled = sg.compile(checkpointer=MemorySaver())
```

The demo supervisor reproduces the loop breaker: if the last message is an `AIMessage` from one of the leaves, it returns `next_agent=None`, which `_research_router` maps to `__end__`; otherwise it classifies the latest human message:

```python
async def demo_supervisor(state: DemoState) -> dict[str, Any]:
    msgs = list(state.get("messages") or [])
    if msgs and getattr(msgs[-1], "type", "") == "ai" \
            and state.get("current_agent") in RESEARCH_AGENTS:
        return {"current_agent": "research_supervisor", "next_agent": None}
    human = next((m for m in reversed(msgs) if getattr(m, "type", "") == "human"), None)
    return {
        "current_agent": "research_supervisor",
        "next_agent": _classify(human.content if human else ""),
    }
```

### The real builder

In the prismal source, `_make_definition()` wires the production `researcher_node` and `rag_agent_node` plus a supervisor from `make_domain_supervisor(domain_name="research", ...)` into a `SubgraphDefinition` (entry point `research_supervisor`, both leaves edging back to it, `_research_router` as the conditional edge). `SubgraphFactory.build()` compiles it with an isolated SQLite checkpointer; `register_research_orchestrator()` installs it idempotently in the global `SubgraphRegistry`, and `get_compiled_research_orchestrator()` returns a process-cached compiled graph. The router is deliberately defensive: any `next_agent` outside `RESEARCH_AGENTS` — including a leaf name from *another* domain — logs a warning and ends the subgraph cleanly.

## Key API

| Symbol | Role |
|---|---|
| `get_compiled_research_orchestrator(...)` | Returns (building if needed) the compiled orchestrator; module-level cache |
| `register_research_orchestrator(...)` | Compiles and registers the orchestrator in `SubgraphRegistry` (idempotent) |
| `_make_definition()` | Builds the `SubgraphDefinition` (entry point `research_supervisor`, 2 leaves, return edges) |
| `make_domain_supervisor("research", ...)` | Produces the LLM router node with built-in loop breaker |
| `_research_router(state)` | Conditional edge: `state["next_agent"]` → leaf name or `__end__` (unknown values end cleanly) |
| `RESEARCH_AGENTS` | `("researcher", "rag_agent")` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/12_research_orchestrator.ipynb
uv run python examples/subgraphs/12_research_orchestrator.py   # from the prismal repo
```

No API keys required — both demos run on deterministic stubs and mocks (the arXiv and MedQuAD CSVs are local, with an embedded fallback).

## Related

- [Analysis Orchestrator](10_analysis_orchestrator.md) — the sibling domain orchestrator for analytical output.
- [Engineering Orchestrator](11_engineering_orchestrator.md) — the sibling domain orchestrator for code and file work.
- [Debate Consensus](08_debate_consensus.md) — another multi-agent subgraph, structured as a debate rather than a router.
