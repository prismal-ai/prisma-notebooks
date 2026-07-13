# Financial Analyst

> **Notebook:** [`notebooks/subgraphs/02_financial_analyst.ipynb`](../../notebooks/subgraphs/02_financial_analyst.ipynb) · **Example:** [`examples/subgraphs/02_financial_analyst.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/02_financial_analyst.py)

The Financial Analyst subgraph turns a ticker into a full research report through five specialist agents: market_data_collector → technical_analyst → fundamental_analyst → risk_sentiment_analyst → report_generator, ending with a BUY/HOLD/SELL recommendation and a price target. What makes it interesting is not the linear chain but the three quality gates spliced between the nodes: a data-availability gate that aborts with a structured error report when the market snapshot is unusable, a technical-confidence gate that skips fundamental analysis when the indicators are too noisy, and an optional human-in-the-loop (HITL) gate before the report is delivered.

Use it as the reference for pipelines where downstream stages must never be asked to analyze empty or unreliable data. The demo analyzes NVDA and MSFT snapshots (Yahoo Finance-style data, embedded).

## What it demonstrates

- The `register_financial_analyst()` + `get_compiled_financial_analyst()` convenience API shared by pipelines 01–03.
- Three quality gates: data availability, technical confidence, and optional HITL approval.
- Graceful degradation paths — a dead-end `insufficient_data_report` node and a `limited_technical_disclaimer` passthrough instead of hard failures.
- Feeding market data through `state["metadata"]` and reading the structured `FinancialReport` back out.
- Runtime-configurable gate thresholds via `Settings` (`financial_min_confidence`, `financial_technical_min_confidence`, `financial_hitl_enabled`).

## Pipeline topology

```
market_data_collector
   │  [Data Gate: missing_fields non-empty AND confidence < threshold?]
   ├── fail ──► insufficient_data_report ──► END
   ▼  pass
technical_analyst
   │  [Technical Gate: data_confidence >= threshold?]
   ├── limited ──► limited_technical_disclaimer ──┐
   ▼  full                                        │
fundamental_analyst ──► risk_sentiment_analyst ◄──┘
                                 │
                                 ▼
                          report_generator
                                 │  [optional HITL Gate: approval_seed → human_approval]
                                 ▼
                                END
```

## How it works

### Registering and compiling the subgraph

The notebook imports the two entry points defensively, then registers and compiles once per session:

```python
from prismal.agents.subgraphs.financial.builder import (
    get_compiled_financial_analyst,
    register_financial_analyst,
)

await register_financial_analyst()
subgraph = await get_compiled_financial_analyst()
```

`register_financial_analyst()` is idempotent and takes an isolated `checkpointer_path` (SQLite); `get_compiled_financial_analyst()` returns the cached compiled graph, building it on first use.

### Feeding the request and market data

Each analysis request carries a ticker, timeframe, and focus, plus a market snapshot in `metadata` so the collector has data to validate:

```python
state = create_initial_state(session_id="nb-financial-analyst")
state["messages"] = [HumanMessage(content=format_analysis_request(request))]
state["metadata"] = {
    "ticker": request["ticker"],
    "market_data": MOCK_MARKET_DATA.get(request["ticker"], {}),
    "hitl_enabled": False,  # disable human approval for the example
}

config = {"configurable": {"thread_id": f"fin_{request['ticker']}_001"}}
final_state = await subgraph.ainvoke(state, config=config)
```

The final report is the last message in `final_state["messages"]`; the structured version lives at `metadata["financial_analyst"]["financial_report"]`.

### Gate 1 — data availability

The gate after `market_data_collector` blocks only when *both* signals fire — an incomplete snapshot **and** low confidence — because a single trip wire would be too trigger-happy:

```python
missing_fields = snapshot.get("missing_fields") or []
confidence_f = float(snapshot.get("data_confidence", 0.0))

unusable = bool(missing_fields) and confidence_f < threshold
if unusable:
    return on_fail   # → insufficient_data_report → END
return on_pass       # → technical_analyst
```

On failure the pipeline does not raise: the `insufficient_data_report` node emits a proper `FinancialReport` with an `Error` section and the standard disclaimer, so dashboards and SDK clients need no separate code path. The threshold defaults to `settings.financial_min_confidence`, read at gate-evaluation time.

### Gate 2 — technical confidence

After `technical_analyst`, a second gate compares `TechnicalAnalysis.data_confidence` against `settings.financial_technical_min_confidence`. A noisy signal does not stop the pipeline — it skips `fundamental_analyst` and routes through a passthrough node that flags the state:

```python
fin["limited_data_disclaimer"] = True
fin["limited_data_note"] = "Technical analysis based on limited data."
```

The `report_generator` picks these flags up and prefixes the final report with the disclaimer, then the flow rejoins at `risk_sentiment_analyst`.

### Gate 3 — optional HITL approval

When `settings.financial_hitl_enabled` is true (env `PRISMAL_FINANCIAL_HITL_ENABLED=true`), the builder splices two extra nodes between `report_generator` and END — `seed_hitl_metadata` (writes the report artifact and `risk_level="MEDIUM"` into metadata) and `human_approval_node` (raises a LangGraph `interrupt()`), followed by an `hitl_gate` conditional edge. With the flag off (the notebook default) `report_generator` wires straight to END, so the demo runs unattended.

## Key API

| Symbol | Role |
|---|---|
| `register_financial_analyst()` | Idempotent async registration in `SubgraphRegistry` |
| `get_compiled_financial_analyst()` | Returns the compiled graph, building it on first call |
| `market_data_collector_node` … `report_generator_node` | The five analyst node functions |
| `_market_data_availability_gate(...)` | Gate 1: missing fields + low confidence → error report |
| `_technical_confidence_gate(...)` | Gate 2: noisy indicators → skip fundamental, add disclaimer |
| `hitl_gate` / `seed_hitl_metadata` / `human_approval_node` | Gate 3: optional human approval of the report |
| `metadata["financial_analyst"]` | `market_snapshot`, `technical_analysis`, `financial_report`, `limited_data_disclaimer` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/02_financial_analyst.ipynb
uv run python examples/subgraphs/02_financial_analyst.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [ML Pipeline](01_ml_pipeline.md) — the same convenience API with a score-based retrain loop.
- [Dev Pipeline](03_dev_pipeline.md) — four gates including the same HITL approval flow.
- [HITL Approval](09_hitl_approval.md) — the human-in-the-loop gate primitives in isolation.
- [Analysis Orchestrator](10_analysis_orchestrator.md) — routes analysis work across subgraphs.
