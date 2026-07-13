# Skynet swarm: end-to-end subgraph demo with injected fakes

> **Notebook:** [`notebooks/skynet_swarm.ipynb`](../notebooks/skynet_swarm.ipynb) · **Example:** [`examples/skynet_swarm.py`](https://github.com/prismal-ai/prismal/blob/main/examples/skynet_swarm.py)

Skynet (Fase S) is prismal's swarm map-reduce layer: a meta-supervisor decomposes one
goal into N independent sub-orders, fans out a dynamically-sized worker swarm with one
LangGraph `Send()` per order, reduces the workers' outputs into a single answer, and
re-plans anything left unmet — bounded, observable, and audited. It is opt-in via
`settings.skynet_enabled`; with the flag off, the compiled supervisor graph is
byte-for-byte unchanged. This notebook builds the full `skynet` subgraph
(`plan → Send fan-out → worker ⇉ reduce → evaluate → output`) and runs a
parallelizable goal through a 3-worker swarm — entirely offline.

The trick that makes the demo deterministic is the framework's factory-injection
pattern: every LLM-backed step (`plan_fn`, `worker_fn`, `evaluate_fn`, `reduce_fn`)
accepts a plain async callable. Inject fakes and the whole pipeline runs with no LLM
and no network, while exercising exactly the same graph topology, state shape, and
security seams a production swarm would. Reach for this pattern when a goal splits
into independent, parallelizable pieces — "research these three competitors" is the
canonical case.

## What it demonstrates

- Assembling the real Skynet topology with `build_skynet_subgraph()` +
  `assemble_state_graph()`, then compiling and invoking it like any LangGraph graph.
- Dynamic swarm sizing: the planner backend decides how many `SwarmOrder`s to emit
  (here, one per competitor); the supervisor hard-caps the count at
  `min(skynet_max_swarm, parallel_max_workers)` and *defers* overflow instead of
  dropping it.
- The map-reduce control loop: plan → one `Send()` per order → parallel workers →
  `reduce_results()` → `evaluate()` → output.
- Where results land: the whole `SwarmResult` (worker results, rounds, completeness)
  under `state["metadata"]["skynet"]["result"]`, and the final answer as the last
  message.
- Zero-dependency testing of a multi-agent architecture via injected fakes.

## How it works

### Setup and goal

The imports pull the value objects (`SwarmOrder`, `SwarmPlan`, `WorkerResult`), the
subgraph builder, and the shared topology assembler. The goal is deliberately
parallelizable — three competitors, three independent research tasks.

```python
from prismal.agents.skynet.types import SwarmOrder, SwarmPlan, WorkerResult
from prismal.agents.subgraphs.factory import assemble_state_graph
from prismal.agents.subgraphs.skynet import build_skynet_subgraph

GOAL = "Research competitors Acme, Globex and Initech in parallel"

_COMPETITORS = ["Acme", "Globex", "Initech"]
```

### The planner fake — dynamic sizing in miniature

`SkynetSupervisor.plan()` calls the injected `plan_fn` with the *secure message list*
built by `SecurePromptBuilder` (system + sanitized user), never a raw f-string of the
goal. The fake returns one `SwarmOrder` per competitor, which is exactly what dynamic
sizing means: the planner — not the graph shape — chooses N. In a real deployment you
omit `plan_fn` and the supervisor lazily wires `ProviderRegistry().get_llm()`,
honouring `settings.skynet_planner_model`.

```python
async def fake_plan(messages: list[dict[str, str]]) -> SwarmPlan:
    return SwarmPlan(
        goal="",
        orders=[
            SwarmOrder(order_id=f"ord-{i}", instruction=f"Research competitor {name}")
            for i, name in enumerate(_COMPETITORS, start=1)
        ],
        rationale="one worker per competitor",
    )
```

After the planner returns, the supervisor caps the order list at the effective
swarm cap; anything beyond it rides on `SwarmPlan.deferred` and is carried into the
next round rather than silently dropped. Each `plan()` call also emits a hash-first
`skynet_plan` audit record — goal hash, mode, requested vs. effective size, deferred
count — never the goal text itself.

### Worker, evaluator, and reducer fakes

Each `SwarmWorker` executes exactly one order. The fake inspects the sanitized user
message (`messages[1]`) to find which competitor it was assigned — mirroring how a
real worker LLM would receive `Order ord-N (attempt 1): Research competitor ...`
through the secure channel:

```python
async def fake_worker(messages: list[dict[str, str]]) -> str:
    user = messages[1]["content"]
    name = next((n for n in _COMPETITORS if n in user), "unknown")
    return f"{name}: strong in EU, weak mobile offering, pricing mid-market."
```

The evaluator decides completeness and synthesizes the current best answer; the
reducer merges successful worker outputs (the injected fake stands in for the default
`synthesis` strategy of `reduce_results()`, which would otherwise call the model —
`concat` and `first_success` are the LLM-free alternatives, selected via
`skynet_reduce_strategy`):

```python
async def fake_evaluate(messages: list[dict[str, str]]) -> tuple[bool, str]:
    return (True, "All three competitors researched: ...")

async def fake_reduce(goal: str, results: list[WorkerResult]) -> str:
    bullet_list = "\n".join(f"- {r.output}" for r in results)
    return f"Competitive landscape:\n{bullet_list}"
```

Because `fake_evaluate` returns `complete=True` on the first round, the conditional
edge after `skynet_evaluate` routes straight to `skynet_output`. Had it returned
`False` with failed orders pending, the loop would route back to `skynet_plan` for a
deterministic re-plan (unmet orders pass through with `attempt + 1`, no LLM call),
bounded by `skynet_max_rounds`.

### Building and running the subgraph

`build_skynet_subgraph()` wires the injected backends into a `SubgraphDefinition`
with five nodes, the `Send` fan-out edge from `skynet_plan` (one `Send` per staged
order, via the reused `make_parallel_dispatcher`), and the bounded re-plan
conditional edge. `assemble_state_graph()` turns that definition into a compilable
`StateGraph[AgentState]`:

```python
definition = build_skynet_subgraph(
    plan_fn=fake_plan,
    worker_fn=fake_worker,
    evaluate_fn=fake_evaluate,
    reduce_fn=fake_reduce,
)
graph = assemble_state_graph(definition).compile()

result = await graph.ainvoke({"messages": [HumanMessage(content=GOAL)]})

swarm_result = result["metadata"]["skynet"]["result"]
```

All Skynet state is namespaced under `state["metadata"]["skynet"]`, keeping the swarm
layer isolated from the rest of `AgentState`. The demo unpacks the frozen
`SwarmResult` — per-worker outputs with success flags, `rounds_completed`,
`complete` — and prints the synthesized answer from the final message. A failed
worker would show up as `WorkerResult(success=False, error=...)`: worker exceptions
are captured inside the node, never raised out of it, so one failure cannot abort the
swarm.

## Key API

| Symbol | Role |
|---|---|
| `build_skynet_subgraph()` | Builds the `SubgraphDefinition` (plan / worker / reduce / evaluate / output) with injected backends |
| `assemble_state_graph()` | Shared sync topology builder: `SubgraphDefinition` → compilable `StateGraph` |
| `SwarmOrder` | Frozen value object: one sub-order (`order_id`, `instruction`, `attempt`) |
| `SwarmPlan` | The planner's decomposition: `orders` (≤ cap), `rationale`, `deferred` overflow |
| `WorkerResult` | One worker's outcome (`output`, `success`, `error`, `tool_calls`) |
| `SwarmResult` | Reduced outcome across rounds, stored at `metadata["skynet"]["result"]` |
| `SkynetSupervisor` | Owns sizing and the control loop (built internally from `plan_fn` / `evaluate_fn`) |

## Run it

```bash
uv run jupyter lab notebooks/skynet_swarm.ipynb
uv run python examples/skynet_swarm.py   # from the prismal repo
```

API key: not required — the demo runs fully offline with injected fakes.

## Related

- [Skynet direct API](skynet_direct_api.md) — the same layer driven without LangGraph, including a re-plan round.
- [Skynet specialist swarm](skynet_specialist_swarm.md) — S+ roles, metering, and a remote A2A worker.
- [Parallel research fan-out](parallel_research_fanout.md) — the underlying `Send()` dispatcher pattern.
- [Budget governance](budget_governance.md) — the `CostMeter`/`BudgetGuard` layer Skynet's token budget builds on.
