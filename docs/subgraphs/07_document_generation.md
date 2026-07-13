# Document Generation

> **Notebook:** [`notebooks/subgraphs/07_document_generation.ipynb`](../../notebooks/subgraphs/07_document_generation.ipynb) · **Example:** [`examples/subgraphs/07_document_generation.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/07_document_generation.py)

The `document_generation` subgraph turns a document request (topic, audience, sections, target length) into a finished technical document through a five-node linear pipeline: **planner → researcher → writer → editor → formatter**. Each node incrementally transforms the artifact — the planner fixes structure and tone, the researcher gathers grounding material (LLM knowledge plus optional RAG), the writer drafts, the editor reviews for clarity and completeness, and the formatter emits the final `markdown`, `plain`, or `html` output.

Use it whenever documents have a predictable structure worth enforcing: API guides, RFCs and ADRs, periodic reports, user manuals, README/CHANGELOG generation. The notebook exercises it against three technical topics (REST API design, ML explainability, zero-trust security) so the value added by each stage is visible.

## What it demonstrates

- A 5-node linear pipeline where every node reads and writes `state["metadata"]["document_generation"]`.
- Structured document requests (title, audience, sections, target length, context) driving the planner.
- An editor stage acting as a lightweight quality gate: missing sections, length shortfall, tone mismatch, absent conclusion.
- Three output formats from one pipeline (`markdown` with YAML frontmatter, `html`, `plain`).
- The standard build pattern: `build_document_generation_subgraph()` returns a `SubgraphDefinition`, compiled via `assemble_state_graph(defn).compile()`.

## Pipeline topology

```
planner ──► researcher ──► writer ──► editor ──► formatter ──► END
                │                                    │
        (optional RAG engine)          (appends AIMessage with final doc)
```

## How it works

### Dataset: structured document requests

Each request fully specifies the document contract that the planner turns into a plan:

```python
DOCUMENT_REQUESTS = [
    {
        "id": "DOC-001",
        "title": "REST API Design Guide",
        "topic": "REST API design best practices",
        "audience": "developers intermediate",
        "format": "markdown",
        "sections": ["Introduction", "Resources and URIs", "HTTP Methods", ...],
        "target_length": "1500 words",
        "context": "Practical guide for backend developers ...",
    },
    # DOC-002: ML Explainability (XAI) · DOC-003: Zero-Trust Security
]
```

A companion `RESEARCH_KNOWLEDGE` dict simulates what the researcher node would extract (key concepts, good/bad examples, references) so the demo runs deterministically without live retrieval.

### Simulated nodes mirror the real pipeline

The notebook implements each node as a plain function to make the stage contracts explicit. The planner derives structure and tone; the editor is the interesting one — it audits the draft and records every fix it applies:

```python
def simulate_editor(draft: str, request: dict) -> tuple[str, list[str]]:
    edits = []
    h2_count = draft.count("\n## ")
    if h2_count < len(request["sections"]) - 1:
        edits.append(f"Added {len(request['sections']) - h2_count} missing sections")
    word_count = len(draft.split())
    target = int(request["target_length"].split()[0])
    if word_count < target * 0.5:
        edits.append(f"Document too short ({word_count} words vs {target} target)")
    if "Conclusion" not in draft:
        # ... append a Conclusion section ...
        edits.append("Added Conclusion section")
    return edited, edits
```

The formatter then applies the requested output format — markdown gains YAML frontmatter, HTML gets a heading/paragraph conversion, plain text strips markup.

### Real mode: build, register, compile, invoke

When prismal is installed, the notebook switches from the simulator to the real LangGraph subgraph. Note the standard pattern: the builder returns a `SubgraphDefinition` (nodes + edges, no compiled graph), and `assemble_state_graph(defn).compile()` produces the runnable graph:

```python
await register_document_generation(format=request["format"])
subgraph = build_document_generation_subgraph(format=request["format"])
graph = assemble_state_graph(subgraph).compile()

state = create_initial_state(session_id="nb-document-generation")
state["messages"] = [HumanMessage(content=f"Generate a technical document about: {request['topic']}\n...")]
state["metadata"] = {"document_generation": {"title": request["title"], ...}}

config = {"configurable": {"thread_id": f"docgen_{request['id']}_001"}}
final_state = await graph.ainvoke(state, config=config)
```

The formatter node also appends an `AIMessage` with the final document, so `final_state["messages"][-1].content` is the finished artifact; word counts and plan data live under `metadata["document_generation"]`.

### Grounding with RAG

Passing a vector store to the builder makes the researcher ground content in your own corpus instead of relying on LLM knowledge alone:

```python
subgraph = build_document_generation_subgraph(
    rag_engine=ChromaVectorStore(collection_name="company_docs"),
    format="markdown",
)
```

## Key API

| Symbol | Role |
|---|---|
| `build_document_generation_subgraph(llm, rag_engine, format, settings)` | Builds the 5-node `SubgraphDefinition` (entry point `planner`, linear edges) |
| `register_document_generation(...)` | Idempotent install into the `SubgraphRegistry` singleton (async) |
| `assemble_state_graph(defn)` | Shared factory that turns a `SubgraphDefinition` into a `StateGraph` ready to `.compile()` |
| `make_planner_node` / `make_researcher_node` / `make_writer_node` / `make_editor_node` | LLM-backed node factories (researcher optionally takes `rag_engine`) |
| `make_formatter_node(format=...)` | Terminal node: applies `markdown \| plain \| html` and appends the final `AIMessage` |
| `metadata["document_generation"]` | Shared state channel every node reads/writes (plan, research, draft, word_count) |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/07_document_generation.ipynb
uv run python examples/subgraphs/07_document_generation.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` for real mode; the simulated pipeline runs offline.

## Related

- [Debate Consensus](08_debate_consensus.md) — another linear 4-node subgraph, synthesizing opposing positions instead of documents.
- [Code Review](04_code_review.md) — pipeline with a report-generation terminal stage and quality scoring.
- [Research Orchestrator](12_research_orchestrator.md) — orchestrates research before writing, at the supervisor level.
- [Customer Service](06_customer_service.md) — pipeline with a conditional escalation gate rather than a purely linear flow.
