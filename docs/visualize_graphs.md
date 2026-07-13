# Visualizing Prismal Graphs as Mermaid Diagrams

> **Notebook:** [`notebooks/visualize_graphs.ipynb`](../notebooks/visualize_graphs.ipynb) · **Example:** [`examples/visualize_graphs.py`](https://github.com/prismal-ai/prismal/blob/main/examples/visualize_graphs.py)

Prismal can render **any graph-based architecture** — a compiled LangGraph, a `SubgraphDefinition`, or a `PrismalStateGraphBuilder` — to a Mermaid `flowchart TD` diagram, entirely offline. This walkthrough builds six reusable subgraph pipelines in memory (customer service, document generation, data ETL, code review, debate consensus, and the multimodal pipeline) and prints the Mermaid source for each one, so you can see the exact topology the framework will execute without running a single node or making a network call.

The visualization helpers ship as part of the developer surface adjacent to the Fase X extension API: they are re-exported from `prismal.langgraph`, the official passthrough module that guarantees version compatibility with the LangGraph release prismal was tested against. Importing `to_mermaid` from there (rather than reaching into `langgraph.*` internals) keeps your code on the stable contract.

## What it demonstrates

- `to_mermaid()` producing Mermaid source for six different `SubgraphDefinition`s — no compilation step needed on your side, no network, no API keys.
- The single coercion path behind all helpers: a `SubgraphDefinition` is assembled via `assemble_state_graph()` and compiled *without a checkpointer*, since visualization only needs the topology.
- The equivalence between the free functions and the instance methods (`to_mermaid(definition)` ≡ `definition.to_mermaid()`).
- The fallback-safe notebook variant `visualize()` and the file-writing variant `save_graph_image()`.
- That non-graph architectures (reasoning patterns like ToT or LATS, modal agents) raise `TypeError` — they are plain classes with no LangGraph topology.

## How it works

### Building the subgraphs in memory

Every builder runs offline: it constructs a `SubgraphDefinition` (nodes, edges, conditional edges, entry point) without touching an LLM, so the whole script is deterministic.

```python
from prismal.agents.subgraphs.code_review.builder import build_code_review_subgraph
from prismal.agents.subgraphs.customer_service.builder import (
    build_customer_service_subgraph,
)
from prismal.agents.subgraphs.data_etl.builder import build_data_etl_subgraph
from prismal.agents.subgraphs.debate_consensus.builder import (
    build_debate_consensus_subgraph,
)
from prismal.agents.subgraphs.document_generation.builder import (
    build_document_generation_subgraph,
)
from prismal.agents.subgraphs.multimodal_pipeline import build_multimodal_subgraph
from prismal.langgraph import to_mermaid

# Each entry is (label, SubgraphDefinition). All builders run offline.
SUBGRAPHS = [
    ("customer_service", build_customer_service_subgraph()),
    ("document_generation", build_document_generation_subgraph()),
    ("data_etl", build_data_etl_subgraph()),
    ("code_review", build_code_review_subgraph()),
    ("debate_consensus", build_debate_consensus_subgraph()),
    ("multimodal_pipeline", build_multimodal_subgraph()),
]
```

Note the import of `to_mermaid` from `prismal.langgraph` — the same module that re-exports `StateGraph`, `START`, `END`, `Send`, `interrupt`, `add_messages`, and a `VERSION` constant resolved from the installed LangGraph distribution.

### Printing each diagram

```python
def main() -> None:
    for label, definition in SUBGRAPHS:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        # Equivalent to definition.to_mermaid().
        print(to_mermaid(definition))
```

Under the hood, `to_mermaid` coerces its argument through one internal helper:

- A **compiled graph** (anything with a callable `get_graph`, including the main supervisor graph or a builder's `compile_raw()` output) is used as-is.
- A **`PrismalStateGraphBuilder`** is first compiled to a `SubgraphDefinition`.
- A **`SubgraphDefinition`** is assembled into a `StateGraph` by the shared sync topology builder `assemble_state_graph()` and compiled without a checkpointer — visualization only needs the shape, not persistence.
- Anything else — a reasoning pattern (`tree_of_thoughts`, `debate`, LATS), a modal agent, an arbitrary object — raises `TypeError`, because there is no LangGraph topology to draw.

The Mermaid text comes from LangGraph's own `get_graph().draw_mermaid()`, so the diagram is guaranteed to match the executable topology, conditional edges included. Paste any block into a Mermaid renderer (GitHub Markdown, mermaid.live, an IDE plugin) to see the flowchart.

### The main supervisor graph

The script ends by pointing at the supervisor-level helper:

```python
print("Main supervisor graph:")
print("  from prismal.agents.graph import visualize_supervisor_graph")
print("  visualize_supervisor_graph()   # builds + draws the compiled graph")
```

`visualize_supervisor_graph()` builds the full SUPERVISOR state machine — the central router plus its 26 specialist agent nodes — and renders it the same way. The demo only prints the instructions because compiling the main graph wires a checkpointer, which is more setup than a topology tour needs.

### Beyond Mermaid text: PNG and notebook rendering

`to_mermaid` never needs a network, but three siblings cover richer outputs:

- `to_mermaid_png(obj)` returns PNG bytes via the LangGraph Mermaid renderer (a mermaid.ink network call or a local engine) — it can fail if neither is available.
- `visualize(obj)` is the fallback-safe notebook variant: it tries an inline PNG through `IPython.display` and, if anything goes wrong (no IPython, no renderer, no network), prints the Mermaid source instead. It never raises for a valid graph.
- `save_graph_image(obj, "graph.png")` writes the PNG to disk (creating parent directories), which is handy for CI artifacts and docs builds.

`SubgraphDefinition` also gains the instance methods `.to_mermaid()`, `.visualize()`, and `.save_image()`, so a definition you hold can render itself without importing the free functions.

## Key API

| Symbol | Role |
|---|---|
| `to_mermaid(obj)` | Mermaid source for any graph-based architecture; fully offline |
| `to_mermaid_png(obj)` | PNG bytes via the LangGraph Mermaid renderer (network or local engine required) |
| `visualize(obj)` | Notebook display with graceful fallback to Mermaid text; never hard-fails on a valid graph |
| `save_graph_image(obj, path)` | Renders to PNG and writes the file (CI / non-notebook use) |
| `SubgraphDefinition.to_mermaid()` / `.visualize()` / `.save_image()` | Instance-method equivalents on any definition |
| `visualize_supervisor_graph()` | Builds and draws the main supervisor graph (`prismal.agents.graph`) |
| `assemble_state_graph()` | Shared sync topology builder that turns a definition into a `StateGraph` |
| `prismal.langgraph` | Version-guaranteed re-export surface (`StateGraph`, `START`, `END`, `Send`, visualization helpers, `VERSION`) |
| `PrismalStateGraphBuilder` | Fluent Fase X builder — also accepted directly by every visualization helper |

## Run it

```bash
uv run jupyter lab notebooks/visualize_graphs.ipynb
uv run python examples/visualize_graphs.py   # from the prismal repo
```

Runs entirely offline — `to_mermaid` needs no network and no API keys (PNG rendering is the only optional online step).

## Related

- [Supervisor quickstart](supervisor_quickstart.md)
- [Custom subgraphs with the extension API](extension/custom_subgraph.md)
- [ML pipeline subgraph](subgraphs/01_ml_pipeline.md)
- [Multimodal pipeline subgraph end-to-end](multimodal_pipeline.md)
