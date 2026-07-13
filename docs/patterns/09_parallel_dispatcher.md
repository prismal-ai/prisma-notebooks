# Parallel Dispatcher

> **Notebook:** [`notebooks/patterns/09_parallel_dispatcher.ipynb`](../../notebooks/patterns/09_parallel_dispatcher.ipynb) · **Example:** [`examples/patterns/09_parallel_dispatcher.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/09_parallel_dispatcher.py)

The Parallel Dispatcher is prismal's map-reduce primitive for LangGraph: `make_parallel_dispatcher()` builds a routing function that reads a list of independent tasks from graph state, emits one LangGraph `Send()` per task targeting the same worker node (fan-out), and lets LangGraph run the workers concurrently and merge their partial state updates back together (fan-in). When a turn contains N independent sub-tasks, running them serially multiplies latency by N for no reason.

The notebook grounds this in fact verification with a sample from FEVER (Fact Extraction and VERification, 185K claims about Wikipedia): six embedded claims ("Python was created by Guido van Rossum…", "The Eiffel Tower is located in Berlin…") that are perfectly independent — the ideal fan-out workload. It demonstrates the same fan-out/fan-in shape twice: first with plain `asyncio.gather` plus a semaphore (parallelism without LangGraph), then with the real `make_parallel_dispatcher` showing the `Send()` objects it emits and its empty-state routing.

## What it demonstrates

- Fan-out/fan-in with `asyncio.gather(..., return_exceptions=True)`: bounded concurrency via `asyncio.Semaphore`, per-task error tolerance, and a serial-vs-parallel timing comparison (~6x estimated speedup for 6 claims).
- Building a LangGraph dispatcher with `make_parallel_dispatcher()` and inspecting the list of `Send(worker_node, {**state, "_task": task})` objects it returns.
- The dispatcher's two safety controls: `max_workers` hard-capped by `settings.parallel_max_workers`, and a global kill switch `settings.parallel_enabled` that routes everything to `on_empty`.
- Graceful degenerate cases: an empty or missing task list routes to `on_empty` (default `"__end__"`) instead of raising.
- A realistic parallel worker: an LLM fact-checker returning a structured SUPPORTS / REFUTES / NOT_ENOUGH_INFO verdict per claim.

## How it works

### The worker: one claim, one verdict

Each parallel unit of work is an ordinary async function. Here it resolves an LLM from the `ProviderRegistry` and classifies one claim, parsing a strict `LABEL:` / `JUSTIFICATION:` reply format:

```python
async def verify_single_claim(claim_task: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    llm = ProviderRegistry(settings=settings).get_llm()

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),  # SUPPORTS / REFUTES / NOT_ENOUGH_INFO
        HumanMessage(content=f"Claim: {claim_task['claim']}"),
    ])
    # ... parse LABEL / JUSTIFICATION from the reply ...
```

Because claims share nothing, workers need no coordination — the definition of a map-reduce-friendly workload.

### Fan-out without LangGraph: `asyncio.gather`

Before touching graphs, the notebook shows the raw pattern. A semaphore bounds concurrency (respecting provider rate limits), and `return_exceptions=True` means one failing claim degrades into an error record instead of sinking the whole batch:

```python
semaphore = asyncio.Semaphore(max_workers)

async def bounded_verify(claim: dict) -> dict[str, Any]:
    async with semaphore:
        return await verify_single_claim(claim)

results = await asyncio.gather(
    *[bounded_verify(claim) for claim in claims],
    return_exceptions=True,
)
```

The timing section measures the parallel wall-clock and estimates the sequential equivalent — with 6 independent LLM calls the speedup approaches the worker count.

### Fan-out inside LangGraph: `make_parallel_dispatcher`

Inside a graph you don't call `gather` yourself — you hand LangGraph a routing function for `add_conditional_edges`, and `Send()` does the fan-out. The factory wires everything declaratively:

```python
dispatcher = make_parallel_dispatcher(
    tasks_field="research_tasks",   # state key holding the task list
    worker_node="claim_verifier",   # graph node invoked once per task
    max_workers=6,                  # per-call cap (hard-capped by settings)
    on_empty="__end__",             # route when there is nothing to dispatch
    task_key="_task",               # where each worker finds its task
)

dispatch_result = dispatcher(mock_state)
# → [Send("claim_verifier", {**state, "_task": {...}}), ...]  one per task
```

Each `Send` copies the parent state and injects the individual task under `task_key`, so the worker node reads `state["_task"]` and otherwise sees normal graph state. LangGraph executes the sends concurrently and merges the workers' partial updates via the reducers declared on `AgentState`. When the task list is empty (or not a list), the dispatcher returns the `on_empty` node name — both `list[Send]` and `str` are valid conditional-edge return types (`DispatcherFn`).

### Operator-grade safety controls

Two settings keep fan-out under operational control, and both live *outside* agent code:

- **`parallel_max_workers`** hard-caps the effective worker count: `effective_max = min(max_workers, settings.parallel_max_workers)`. A caller cannot exceed the operator-configured ceiling; excess tasks are dropped with a `parallel_dispatcher_truncated` warning.
- **`parallel_enabled = False`** makes every dispatch route to `on_empty`, disabling parallelism globally — invaluable for debugging race-like behaviour without touching graph definitions.

The notebook prints both settings up front and demonstrates the empty-state route, so you see all the dispatcher's non-fan-out branches, not just the happy path.

### Fan-in: tolerating partial failure

Aggregation is where parallel pipelines usually break: one exception in a batch of six should not discard the other five verdicts. The `gather` demo turns exceptions into structured error records during the fan-in pass, so the summary table can show per-claim status honestly:

```python
valid_results = []
for i, result in enumerate(results):
    if isinstance(result, Exception):
        valid_results.append({"claim_id": claims[i]["id"],
                              "error": str(result), "correct": False})
    else:
        valid_results.append(result)
```

Inside a LangGraph graph the equivalent resilience comes from the reducers: each worker contributes a partial state update, and a worker that fails simply contributes nothing — the merged state still carries every successful result. The notebook ends by printing accuracy against the expected FEVER labels and a sample of the model's justifications.

## Key API

| Symbol | Role |
|---|---|
| `make_parallel_dispatcher(tasks_field, worker_node, max_workers=10, on_empty="__end__", task_key="_task")` | Factory returning a LangGraph conditional-edge callable that fans tasks out as `Send` objects; validates its arguments (`max_workers ≥ 1`, non-empty names). |
| `DispatcherFn` | Type alias of the returned callable: `(AgentState) -> list[Send] \| str`. |
| LangGraph `Send` | The fan-out primitive (from `langgraph.types`): one per task, targeting `worker_node` with the task injected under `task_key`. |
| `get_settings` | Supplies the safety controls: `parallel_max_workers` (hard cap) and `parallel_enabled` (global kill switch). |
| `ProviderRegistry` | Resolves the LLM used by the fact-checking worker in the `asyncio.gather` demo. |

## Run it

```bash
uv run jupyter lab notebooks/patterns/09_parallel_dispatcher.ipynb
# or the script version:
uv run python examples/patterns/09_parallel_dispatcher.py   # from the prismal repo
```

No API key is required for the `make_parallel_dispatcher` demo itself; the fact-verification benchmark makes real LLM calls, so set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env` to run it end to end.

## Related

- [LLM Compiler](04_llm_compiler.md) — DAG-based parallelism: when tasks have *dependencies*, the compiler schedules waves instead of a flat fan-out.
- [Parallel research fan-out](../parallel_research_fanout.md) — this dispatcher powering the `parallel_research` route in the full supervisor graph.
- [Skynet swarm](../skynet_swarm.md) — a planner-driven swarm built on the same `Send` fan-out, with re-planning and reduction strategies on top.
- [Budget governance](../budget_governance.md) — fan-out multiplies LLM calls; budget guards keep the total spend bounded.
