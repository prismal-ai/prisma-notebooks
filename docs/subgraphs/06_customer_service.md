# Customer Service

> **Notebook:** [`notebooks/subgraphs/06_customer_service.ipynb`](../../notebooks/subgraphs/06_customer_service.ipynb) · **Example:** [`examples/subgraphs/06_customer_service.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/06_customer_service.py)

The `customer_service` subgraph automates first-line support with a confidence-gated escalation path. A `classifier` categorizes the incoming query (faq / complaint / technical / other), `faq_retrieval` runs RAG over an internal knowledge base, and an `escalation_gate` then branches: high-confidence answers go to `response_generator` (an auto-reply to the customer), while complaints, low-confidence hits, or queries with missing metadata go to `ticket_creator` (a ticket for a human agent).

The demo processes eight support queries in the style of ATIS intents and Amazon customer reviews — clean FAQ questions, technical issues, angry complaints, and an ambiguous exception request — and verifies that each one takes the expected branch. Use this subgraph when you have a well-curated knowledge base and a high volume of repetitive queries, and you want deterministic, auditable escalation rather than an LLM improvising on hard cases.

## What it demonstrates

- The escalation gate: a conditional-edge function from `make_escalation_gate(threshold=0.6)` that routes on category and confidence, escalating by default when metadata is missing (safer to open a ticket than hallucinate).
- Intent classification and RAG retrieval feeding a single routing decision (`route`, recorded in metadata).
- Building via `build_customer_service_subgraph()` → `SubgraphDefinition` → `assemble_state_graph(defn).compile()`.
- How `escalation_threshold` shifts the balance between auto-resolution and human tickets.
- Why the pipeline always escalates without a knowledge base: `rag_engine=None` short-circuits retrieval with zero confidence.

## Pipeline topology

```
classifier → faq_retrieval → escalation_gate ──┬──► response_generator   (confidence ≥ threshold)
                                               └──► ticket_creator       (complaint | low confidence | missing metadata)
```

In the builder, `escalation_gate` is a pass-through node; the actual routing happens in the conditional edge attached to it.

## How it works

### Dataset and knowledge base

Eight `SUPPORT_QUERIES` each carry an `expected_route` so routing accuracy can be scored: FAQ and technical queries should reach `response_generator`; complaints and the ambiguous exception request should reach `ticket_creator`. The `FAQ_KB` holds five entries (returns, shipping, support hours, device troubleshooting, password recovery) — in production this would be indexed in a vector store and served through the `rag_engine`.

### Classification and retrieval

The notebook simulates the two upstream nodes with deterministic heuristics (in real mode the classifier is an LLM and retrieval is RAG). Note how complaints are deliberately assigned low confidence so the gate escalates them:

```python
def classify_intent(query: str) -> tuple[str, float]:
    """Simulate the classifier node (LLM in real mode)."""
    # ... count complaint / faq / technical keyword signals
    if complaint_score >= 2:
        return "complaint", 0.30  # low confidence → will escalate
    if complaint_score == 1:
        return "complaint", 0.45  # still low → will escalate
    if technical_score >= 1:
        return "technical", 0.75
    if faq_score >= 1:
        return "faq", 0.85
    return "other", 0.40  # ambiguous → will escalate
```

`faq_retrieval` scores keyword overlap against the KB as a proxy for semantic similarity, and the two confidences are averaged into the value the gate inspects.

### The escalation gate

The real gate (in `escalation_node.py`) is a plain function from state to the next node name. It escalates on three conditions, in order — missing metadata, complaint category, low confidence:

```python
def gate(state: dict[str, Any]) -> str:
    cs = (state.get("metadata") or {}).get("customer_service") or {}
    category = cs.get("category")
    confidence = cs.get("confidence")
    if category is None or confidence is None:
        return "ticket_creator"          # defensive default
    if category == "complaint":
        return "ticket_creator"          # complaints always get a human
    if float(confidence) < threshold:
        return "ticket_creator"          # not confident enough to auto-reply
    return "response_generator"
```

The demo mirrors this: complaints escalate regardless of score, and everything else is compared against the 0.6 threshold. Escalated queries get a ticket ID and an acknowledgement message; the rest get a generated response grounded in the retrieved FAQ answer.

### Build, compile, invoke

Unlike the code-review and ETL subgraphs, the builder here takes infrastructure rather than per-node callables: a `rag_engine` for retrieval and an `llm` for the classifier and response generator (defaulting to `ProviderRegistry().get_llm()`, which is why an API key is required). The definition is compiled with the shared factory:

```python
await register_customer_service(escalation_threshold=escalation_threshold)
subgraph = build_customer_service_subgraph(
    escalation_threshold=escalation_threshold,
)
graph = assemble_state_graph(subgraph).compile()

state = create_initial_state(session_id="nb-customer-service")
state["messages"] = [HumanMessage(content=query)]
state["metadata"] = {
    "customer_service": {"user_id": query_data["user_id"], "query_id": query_data["id"]}
}

config = {"configurable": {"thread_id": f"cs_{query_data['id']}_001"}}
final_state = await graph.ainvoke(state, config=config)

cs_meta = final_state.get("metadata", {}).get("customer_service", {})
route = cs_meta.get("route")          # "response_generator" | "ticket_creator"
ticket_id = cs_meta.get("ticket_id")  # set only on the escalation branch
```

`build_customer_service_subgraph()` returns a `SubgraphDefinition`; `assemble_state_graph(defn).compile()` produces the runnable graph.

### Tuning the threshold

The final section sweeps `escalation_threshold` to show its operational meaning: at 0.3 almost everything auto-resolves, 0.6 is the balanced default, 0.8 escalates aggressively, and 1.0 sends every query to a human. The demo also prints routing accuracy against the `expected_route` labels — with the caveat that without a knowledge base RAG confidence is 0.0 and every query escalates.

## Key API

| Symbol | Role |
|---|---|
| `build_customer_service_subgraph(rag_engine=None, escalation_threshold=0.6, llm=None)` | Builds the `SubgraphDefinition` (5 nodes, conditional edge on the gate) |
| `register_customer_service(...)` | Idempotent registration in `SubgraphRegistry` |
| `assemble_state_graph(defn)` | Compiles the definition into a runnable `StateGraph` |
| `make_classifier_node(llm)` | LLM intent classification → `category`, `confidence` in metadata |
| `make_faq_retrieval_node(rag_engine)` | RAG over the KB; `rag_engine=None` short-circuits → always escalates |
| `make_escalation_gate(threshold)` | Conditional-edge fn: complaint / low confidence / missing metadata → `ticket_creator` |
| `make_response_generator_node(llm)` / `make_ticket_creator_node()` | The two terminal branches |
| `state["metadata"]["customer_service"]` | Carries `user_id`, `query_id`, `category`, `confidence`, `route`, `ticket_id` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/06_customer_service.ipynb
uv run python examples/subgraphs/06_customer_service.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [Data ETL + EDA](05_data_etl.md) — the other gated pipeline in this series, branching on validation instead of confidence.
- [Document Generation](07_document_generation.md) — planner → researcher → writer, another linear content pipeline.
- [HITL Approval](09_hitl_approval.md) — pausing a graph for a human decision via `interrupt()`, the complement to automatic escalation.
