# Observability Integration: One Port for Traces, Scores, and Datasets

> **Notebook:** [`notebooks/observability_integration.ipynb`](../notebooks/observability_integration.ipynb) · **Example:** [`examples/observability_integration.py`](https://github.com/prismal-ai/prismal/blob/main/examples/observability_integration.py)

prismal already *observes* a lot — OpenTelemetry spans, Langfuse traces, structlog events live in `prismal/monitoring/`. What Phase OBS adds is a single vendor-neutral seam over all of it: the `ObservabilityPort` protocol. Any object with four methods — `record_node`, `record_score`, `get_run_summary`, `export_dataset` — can receive the full telemetry of a supervisor run: which nodes were visited, which tools were called, how many tokens each hop consumed, and what quality scores (human or LLM-judge) were attached afterwards. The port carries **structural data only** — node names, durations, token counts — never raw prompt or response content, so telemetry cannot become a data-exfiltration channel.

This notebook drives the whole surface with a `FakeObservabilityProvider`, prismal's in-memory implementation of the port: no OTel collector, no Langfuse project, no LLM, no network. It is the same double the framework's own tests inject, which makes this a faithful tour of the contract rather than a mock-heavy approximation. In production you would seed `DefaultObservabilityProvider` instead, which forwards the identical calls to the existing `OTelManager` and `LangfuseManager` singletons.

## What it demonstrates

- The shared naming convention — `run_name_for()` and `trace_tags_for()` — that both LangSmith- and Langfuse-style dashboards group runs by.
- Seeding a per-run provider through `seed_observability_run()`, an in-process registry keyed by `session_id` so the live provider never enters checkpointed state.
- Recording node visits, a tool call, and LLM `Usage` with `record_node()`, then reading the accumulated `RunSummary` back.
- Attaching an LLM-judge score with `record_score()` — the exact hook the Phase V eval harness plugs into.
- Exporting the run as both a LangSmith-shaped and a Langfuse-shaped evaluation-dataset record via `export_dataset()`.

## How it works

### A naming convention both dashboards share

Everything starts with deterministic, PII-free names. `run_name_for()` is the single source of truth for run identifiers (`"{agent}.{session}.turn{n}"`), and `trace_tags_for()` produces the `agent:` / `node:` / `org:` tag list used for Langfuse tags and LangSmith run metadata alike:

```python
session_id = "sess-demo"

print("run name:", run_name_for(agent_name="planner", session_id=session_id, turn=0))
# -> planner.sess-demo.turn0
print("tags:", trace_tags_for(agent_name="planner", node="coder", org_id="acme"))
# -> ['agent:planner', 'node:coder', 'org:acme']
```

Because the run id embeds the agent name as its first dotted segment, downstream consumers can recover the agent from an id alone — the default provider uses exactly this to label its OTel counters.

### Seeding a per-run provider outside checkpointed state

A live provider holds in-memory span buffers (and, in the default implementation, references to OTel and Langfuse clients) — none of which may ever reach LangGraph's checkpoint serializer. So, mirroring `budget/resolve.py`, Phase OBS keeps the provider in an in-process registry keyed by `session_id`; the checkpointed `AgentState` carries only a serializable marker (`{"enabled": True, "run_id": ...}`):

```python
provider = FakeObservabilityProvider()
run_id = seed_observability_run(session_id, provider, agent_name="planner", turn=0)
resolved = get_observability_provider(session_id)
assert resolved is provider
```

Seeding is idempotent per `(session_id, turn)`: repeat calls within a turn return the same `run_id` and keep the seeded provider, while a new turn replaces the entry and starts a fresh run — the same turn-scope convention the budget layer uses. The `finally: clear_observability_run(session_id)` at the end of the demo is the release half of that lifecycle.

### Recording node visits and tool calls

The demo then plays supervisor, recording two node hops. Each `record_node()` call carries the node's status, wall-clock duration, any tool calls it made (`ToolCallRecord`), and its LLM usage as a `prismal.budget.types.Usage` — the same summable value object the budget layer meters with:

```python
resolved.record_node(
    run_id=run_id,
    node_name="coder",
    session_id=session_id,
    status="ok",
    duration_ms=12.0,
    tool_calls=[ToolCallRecord(tool_name="write_file", node="coder", ok=True)],
    usage=Usage(prompt_tokens=15, completion_tokens=25, calls=1),
)

summary = resolved.get_run_summary(run_id)
print("visited:", summary.visited_nodes)          # ['planner', 'coder']
print("tokens:", summary.usage.total_tokens)      # 70 — Usage folds with `+`
```

The port contract makes `record_node` sync and **fail-open**: a backend error is logged, never propagated into the graph. Telemetry must never take a run down. The `RunSummary` snapshot accumulates `visited_nodes`, `spans`, `tool_calls`, folded `usage`, and total `latency_ms` per run; the default provider additionally caps buffers (`observability_run_buffer_size`) and LRU-evicts old runs (`observability_max_runs`).

### Scores and dataset export

Quality signals attach to the same run id. `record_score()` appends a named `ScoreAnnotation` with a `source` of `human`, `llm_judge`, or `system` — this is precisely where the Phase V eval harness (see the agent-eval article) writes its judge verdicts, and where the default provider also mirrors the score to Langfuse as trace feedback:

```python
resolved.record_score(
    run_id=run_id,
    name="llm_judge:groundedness",
    value=0.82,
    source="llm_judge",
)
```

Finally, `export_dataset()` turns recorded runs into evaluation-dataset records in either vendor's shape — snake_case `inputs`/`outputs`/`reference_outputs` for LangSmith, camelCase `input`/`expectedOutput`/`metadata` for Langfuse. The fake (like the default provider) carries no prompt/response content, so the payload fields are empty and only the structural metadata (`run_id`, `agent_name`, `session_id`) is populated. Unknown run ids are skipped, never raised on.

### From fake to live

Swapping the fake for production is a one-line change at the seeding site:

```python
provider = DefaultObservabilityProvider()          # instead of FakeObservabilityProvider()
run_id = seed_observability_run(session_id, provider, agent_name="planner", turn=0)
```

Every `record_node` now also lands in OTel counters, every `record_score` is mirrored to Langfuse trace feedback, and buffer/eviction caps kick in — while the calling code, and anything checkpointed, stays identical. The whole layer is gated by `settings.observability_enabled` (default `False`): with the flag off, `RuntimeContext.observability` is `None` and the pre-existing OTel/Langfuse call sites behave exactly as before.

## Key API

| Symbol | Role |
|---|---|
| `ObservabilityPort` | `@runtime_checkable` protocol: `record_node` / `record_score` / `get_run_summary` / `export_dataset`; sync, fail-open |
| `FakeObservabilityProvider` | In-memory, unbounded implementation — the deterministic test double this demo drives |
| `DefaultObservabilityProvider` | Production adapter forwarding to the existing `OTelManager` + `LangfuseManager` (glue, not a new backend) |
| `run_name_for()` / `trace_tags_for()` | Deterministic, PII-free naming convention shared by both dashboards |
| `seed_observability_run()` / `get_observability_provider()` / `clear_observability_run()` | Per-run provider registry keyed by `session_id`; idempotent per turn, never checkpointed |
| `RunSummary` / `ToolCallRecord` / `ScoreAnnotation` | Frozen value objects: accumulated run snapshot, one tool invocation, one attached score |
| `Usage` (from `prismal.budget.types`) | Summable token/call counter reused from the budget layer |
| `DatasetFormat` | `LANGSMITH` \| `LANGFUSE` — selects the export record shape |
| `Settings.observability_*` | `observability_enabled` (opt-in, default `False`), run buffer size, max runs, default score source and export format |

## Run it

```bash
uv run jupyter lab notebooks/observability_integration.ipynb
uv run python examples/observability_integration.py   # from the prismal repo
```

No API key is required — the demo runs entirely offline against the injected fake. The notebook's final run cell ships commented out as `# main()`; since `main` is synchronous here, just uncomment it and run — no `await` needed.

## Related

- [Agent evaluation harness](agent_eval.md) — the Phase V harness whose LLM-judge scores land in `record_score()`
- [Cost & budget governance](budget_governance.md) — the `Usage` / per-run-registry conventions this layer mirrors
- [Runtime hardening](runtime_hardening.md) — the enforcement-side counterpart to this observation layer
- [Supervisor quickstart](supervisor_quickstart.md) — the graph whose node visits these runs record
