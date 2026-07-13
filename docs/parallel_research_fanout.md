# Parallel research fan-out / fan-in

> **Notebook:** [`notebooks/parallel_research_fanout.ipynb`](../notebooks/parallel_research_fanout.ipynb) · **Example:** [`examples/parallel_research_fanout.py`](https://github.com/prismal-ai/prismal/blob/main/examples/parallel_research_fanout.py)

Phase 34 gave the supervisor a map-reduce branch: when a turn routes to `parallel_researcher`, a
dispatcher fans out one LangGraph `Send()` per pending sub-task to a worker node, the workers run
concurrently, and their partial results converge on an aggregator that builds a single markdown
summary before handing control back to the supervisor. This notebook rebuilds that exact branch as
a small standalone `StateGraph` (imported through `prismal.langgraph`, the framework's official
LangGraph re-export surface), reusing the *real* anchor and aggregator nodes but swapping the
LLM-backed worker for a deterministic fake — so the complete fan-out/fan-in mechanics run with no
provider keys and no network.

Use this pattern when one request decomposes into independent sub-questions that can be researched
in parallel and merged: the sequential `researcher` handles one query per turn, while this branch
bounds latency by the slowest worker rather than the sum of all of them.

## What it demonstrates

- The Phase 34 topology: `START → parallel_researcher → [one Send per task] → worker (×N) →
  research_aggregator → END`, mirroring `prismal/agents/graph.py`.
- `make_parallel_dispatcher` — the same factory that builds the production `research_dispatcher` —
  used as a conditional-edge function that emits `Send` objects.
- Fan-in via the `operator.add` reducer on `state["parallel_results"]`: every concurrent worker
  invocation appends to the same merged list.
- The real `research_aggregator_node` consolidating merged results into one `AIMessage` markdown
  summary.
- The empty-task path: with no `pending_tasks`, the dispatcher routes straight to its `on_empty`
  target instead of raising.

## How it works

### A deterministic worker in place of the ReAct loop

The production worker, `parallel_researcher_worker`, reads its assigned sub-task from
`state["_task"]` (populated by the dispatcher, which copies the parent state into each `Send`
payload), runs a bounded ReAct loop with the researcher's injected tools, and appends one result
dict to `parallel_results`. The notebook keeps that contract but derives a canned finding instead:

```python
async def fake_research_worker(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for ``parallel_researcher_worker``."""
    task = str(state.get("_task", ""))
    return {
        "parallel_results": [
            {
                "agent": "fake_research_worker",
                "task": task,
                "result": f"Deterministic findings for: {task.lower()}",
            }
        ]
    }
```

Because `parallel_results` is declared with an `operator.add` reducer on `AgentState`, LangGraph
merges each worker's single-element list into one combined list at fan-in — no locking, no manual
gathering.

### Wiring dispatcher, workers, and aggregator

`make_parallel_dispatcher` returns a synchronous callable suitable for
`add_conditional_edges`. It reads `state[tasks_field]` and emits one `Send(worker_node, payload)`
per task, capping the fan-out at `max_workers` (itself hard-capped by
`settings.parallel_max_workers`); when the list is missing or empty — or when
`settings.parallel_enabled` is off — it returns the `on_empty` node name instead:

```python
def build_fanout_graph() -> Any:
    dispatcher = make_parallel_dispatcher(
        tasks_field="pending_tasks",
        worker_node="fake_research_worker",
        max_workers=5,
        on_empty="research_aggregator",
    )

    builder = StateGraph(AgentState)
    builder.add_node("parallel_researcher", parallel_researcher_node)
    builder.add_node("fake_research_worker", fake_research_worker)
    builder.add_node("research_aggregator", research_aggregator_node)

    builder.add_edge(START, "parallel_researcher")
    builder.add_conditional_edges(
        "parallel_researcher",
        dispatcher,
        {
            "fake_research_worker": "fake_research_worker",
            "research_aggregator": "research_aggregator",
        },
    )
    builder.add_edge("fake_research_worker", "research_aggregator")
    builder.add_edge("research_aggregator", END)
    return builder.compile()
```

Two of the three nodes are the real framework code. `parallel_researcher_node` is a pass-through
anchor — LangGraph requires conditional edges to originate from a registered node, so it only sets
`current_agent` and lets the dispatcher attached to its conditional edges do the actual fan-out.
`research_aggregator_node` runs once after every worker finishes (LangGraph synchronizes fan-in
automatically when Send-fanned edges converge on a single downstream node) and builds a markdown
summary listing each task and its findings as a new `AIMessage`. In the production graph that
aggregator routes back to the supervisor; here it routes to `END`.

### Running the fan-out — and the empty path

The demo seeds three research sub-tasks and invokes the compiled graph once:

```python
state = create_initial_state(session_id="fanout-demo")
state["messages"] = [HumanMessage(content="Research the RAG stack options.")]
state["pending_tasks"] = list(TASKS)

result = await graph.ainvoke(state)

for item in result["parallel_results"]:
    print(f"  - [{item['agent']}] {item['task']!r}")
print(result["messages"][-1].content)   # aggregated markdown summary
```

All three workers execute concurrently within the same superstep; the merged
`parallel_results` list carries one entry per task, and the last message is the aggregator's
"Parallel research summary". A second invocation with no `pending_tasks` shows the degenerate
case: the dispatcher takes its `on_empty` route directly to the aggregator, which reports that
parallel research produced no results — the graph terminates cleanly rather than erroring.

## Key API

| Symbol | Role |
|---|---|
| `prismal.agents.patterns.parallel.make_parallel_dispatcher` | Factory for a `Send()`-emitting conditional-edge dispatcher (`tasks_field`, `worker_node`, `max_workers`, `on_empty`) |
| `prismal.agents.parallel_research.parallel_researcher_node` | Pass-through anchor node the dispatcher's conditional edges originate from |
| `prismal.agents.parallel_research.research_aggregator_node` | Fan-in node: consolidates `parallel_results` into one markdown `AIMessage` |
| `prismal.langgraph.StateGraph` / `START` / `END` | Official re-export of the LangGraph build surface, version-matched to prismal |
| `prismal.langgraph.create_initial_state` / `AgentState` | The shared state schema whose `parallel_results` field uses an `operator.add` reducer |

## Run it

```bash
uv run jupyter lab notebooks/parallel_research_fanout.ipynb
uv run python examples/parallel_research_fanout.py   # from the prismal repo
```

No API key required — the fake worker keeps the whole fan-out offline.

## Related

- [Supervisor quickstart](supervisor_quickstart.md) — the state machine this branch is grafted onto
- [Hierarchical supervisor](hierarchical_supervisor.md) — scaling routing instead of execution
- [Skynet swarm](skynet_swarm.md) — goal decomposition plus Send()-based worker swarms with re-planning
- [Budget governance](budget_governance.md) — capping the cost of fan-out patterns
