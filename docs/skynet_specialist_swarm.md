# Skynet specialist swarm: roles, metering, and a remote A2A worker

> **Notebook:** [`notebooks/skynet_specialist_swarm.ipynb`](../notebooks/skynet_specialist_swarm.ipynb) · **Example:** [`examples/skynet_specialist_swarm.py`](https://github.com/prismal-ai/prismal/blob/main/examples/skynet_specialist_swarm.py)

Phase S gives prismal a homogeneous swarm: every worker is interchangeable, and the
supervisor's only job is to size, dispatch, reduce, and re-plan. Phase S+ — the
subject of this notebook — layers *heterogeneity* on top without changing the graph
shape: a `RoleRegistry` binds each `SwarmOrder.role` to a `SpecialistRole` profile
(per-role model, tool capabilities, system persona, or a remote A2A agent), a shared
`CostMeter` totals the whole swarm's token usage onto `SwarmResult.usage`, and a role
whose `remote_agent` is set has its order delegated over A2A instead of run
in-process. Specialization is data, not topology — the swarm graph stays identical
and fully testable.

The demo builds the Skynet subgraph with two roles — a local `researcher` specialist
and a `legal` role bound to a remote agent card URL — plus a metering LLM stand-in
and an in-process `send_fn`, so everything runs with no real LLM and no network while
exercising the genuine S+ code paths.

## What it demonstrates

- Declaring specialist profiles with `RoleRegistry` / `SpecialistRole`: a per-role
  model, capabilities, and persona for local workers, and a `remote_agent` card URL
  for A2A delegation.
- The planner tagging each sub-order with a `role`; the supervisor validates tags
  against the registry's known roles (an unknown tag falls back to `"worker"` rather
  than aborting the swarm).
- Whole-swarm cost metering: one shared `CostMeter` threaded through supervisor,
  workers, and reducer, surfaced as `SwarmResult.usage` (`total_tokens`, `calls`).
- Remote delegation over A2A via an injected `send_fn`, with `WorkerResult.remote`
  distinguishing remote from local execution.
- The opt-in gates: `skynet_specialists_enabled`, `skynet_remote_workers_enabled`,
  and `a2a_enabled` — all default-off, preserving Phase-S behaviour byte-for-byte.

## How it works

### A metering LLM stand-in

To show real usage accounting without a provider, the demo patches
`ProviderRegistry` with a fake whose responses carry LangChain-style
`usage_metadata`. The worker's default backend extracts those tokens, prices the
call, and records it into the shared meter — exactly the path a real response takes:

```python
class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 40, "output_tokens": 15}


class MeteringProviderRegistry:
    def get_llm(self, *, model: str | None = None) -> Any:
        class _LLM:
            async def ainvoke(self, _messages: list[Any]) -> _FakeResponse:
                return _FakeResponse(f"[{model or 'default'}] market looks favorable")

        return _LLM()
```

Note the `model` echo: it proves the `researcher` order was executed with the role's
own model override (`research-model`), not the global `skynet_worker_model`.

### Role-tagged planning

The planner fake tags each order with a specialist role. In a real deployment the
supervisor extends the planning system prompt with the registry's known roles and
asks the model to pick the best fit; here the fake pins the assignment. Either way,
the goal reaches the planner only through `SecurePromptBuilder`, and `plan()` writes
a hash-first `skynet_plan` audit record:

```python
async def fake_plan(_messages: list[dict[str, str]]) -> SwarmPlan:
    return SwarmPlan(
        goal="",
        orders=[
            SwarmOrder(order_id="ord-1", instruction="Research the market", role="researcher"),
            SwarmOrder(order_id="ord-2", instruction="Review the plan legally", role="legal"),
        ],
        rationale="one specialist per sub-order",
    )
```

The usual sizing rules still apply: the order list is hard-capped at
`min(skynet_max_swarm, parallel_max_workers)` with overflow deferred to the next
round, and the fan-out is one LangGraph `Send()` per staged order.

### The role registry — one local, one remote

```python
def _registry() -> RoleRegistry:
    return RoleRegistry(
        {
            "researcher": SpecialistRole(
                name="researcher",
                model="research-model",
                capabilities=["research", "web"],
                persona="You are a meticulous market analyst.",
            ),
            "legal": SpecialistRole(
                name="legal",
                remote_agent="https://legal.example.com/.well-known/agent-card.json",
            ),
        }
    )
```

For a local role, the worker prepends the persona to its trusted system template
(never f-string-injected into the user channel), resolves tools through the injected
`ToolProviderPort` filtered by the role's `capabilities`, and calls the role's model.
For a remote role, the worker skips all of that and delegates the entire order via
`send_fn`; the fake here answers in-process, standing in for a real `A2AClient` call.
`RoleRegistry.resolve()` never raises — an unknown role degrades to the generic
`DEFAULT_ROLE` worker — and a remote failure is contained as
`WorkerResult(success=False, remote=True)` so the rest of the swarm still reduces.

### Enabling S+ and running the swarm

The S+ features are opt-in, so the demo constructs a `Settings` with all three flags
on and threads the registry, the remote backend, and the fakes into the builder. One
`CostMeter` is created inside `build_skynet_subgraph()` and shared by the
supervisor, every worker, and the reducer, so `SwarmResult.usage` is the whole-swarm
total (also the seam where `skynet_token_budget` is enforced — breaching it raises
`SkynetBudgetExceeded` at the round boundary):

```python
settings = Settings(
    _env_file=None,
    skynet_specialists_enabled=True,
    skynet_remote_workers_enabled=True,
    a2a_enabled=True,
)
definition = build_skynet_subgraph(
    settings=settings,
    plan_fn=fake_plan,
    evaluate_fn=fake_evaluate,
    reduce_fn=fake_reduce,
    role_registry=_registry(),
    send_fn=fake_send,
)
graph = assemble_state_graph(definition).compile()
result = await graph.ainvoke({"messages": [HumanMessage(content=GOAL)]})
```

The output inspects `result["metadata"]["skynet"]["result"]`: each `WorkerResult`
carries its `role` and a `remote` flag (`ord-1 / researcher / local`,
`ord-2 / legal / remote`), and `swarm_result.usage` reports the metered tokens and
call count — here only the local researcher's single metered call, since the remote
worker consumed no local tokens and the plan/evaluate/reduce backends are injected
fakes that own their own (non-)metering.

## Key API

| Symbol | Role |
|---|---|
| `RoleRegistry` | Maps `SwarmOrder.role` → `SpecialistRole`; `resolve()` never raises (unknown → `DEFAULT_ROLE`) |
| `SpecialistRole` | Frozen profile: `model`, `capabilities`, `persona`, optional `remote_agent` (A2A card URL) |
| `build_skynet_subgraph(settings=..., role_registry=..., send_fn=...)` | Threads S+ collaborators into supervisor and worker |
| `SwarmOrder.role` | The planner's specialist tag, validated against the registry's known roles |
| `WorkerResult.role` / `.remote` | Which role ran the order and whether it was delegated over A2A |
| `SwarmResult.usage` | Whole-swarm `Usage` total from the shared `CostMeter` |
| Settings flags | `skynet_specialists_enabled`, `skynet_remote_workers_enabled`, `a2a_enabled` (all default `False`) |

## Run it

```bash
uv run jupyter lab notebooks/skynet_specialist_swarm.ipynb
uv run python examples/skynet_specialist_swarm.py   # from the prismal repo
```

API key: not required — the demo runs fully offline with injected fakes.

## Related

- [Skynet swarm](skynet_swarm.md) — the baseline Phase-S subgraph this builds on.
- [Skynet direct API](skynet_direct_api.md) — supervisor, worker, and reducer driven without LangGraph.
- [Budget governance](budget_governance.md) — the `CostMeter` and `Usage` machinery behind swarm metering.
- [Custom tool provider](tool_provider_custom.md) — how workers resolve capability-filtered tools via `ToolProviderPort`.
