# Skynet direct API: plan, workers, reduce, evaluate — no subgraph

> **Notebook:** [`notebooks/skynet_direct_api.ipynb`](../notebooks/skynet_direct_api.ipynb) · **Example:** [`examples/skynet_direct_api.py`](https://github.com/prismal-ai/prismal/blob/main/examples/skynet_direct_api.py)

The Skynet subgraph (Fase S, prismal's swarm map-reduce layer) is really a thin
LangGraph shell around three composable pieces: `SkynetSupervisor` (sizing, planning,
evaluation), `SwarmWorker` (one order in, one `WorkerResult` out), and
`reduce_results()` (merging outputs). This notebook drives those pieces *directly* —
no LangGraph, no `Send()`, just `asyncio.gather` — which makes the control loop's
mechanics visible: what a plan looks like, where the hard cap and deferred overflow
live, how a worker failure is contained, and what a deterministic re-plan round
actually does.

The demo deliberately makes one order fail on its first attempt, so you see the full
two-round story: plan → fan out → reduce → evaluate (incomplete) → re-plan the single
unmet order (pass-through with `attempt + 1`, the planner backend is *not* consulted)
→ retry → reduce and evaluate again. Use this style when embedding the swarm layer
in your own orchestration, or when unit-testing swarm behaviour without compiling a
graph.

## What it demonstrates

- Calling `SkynetSupervisor.plan()` / `evaluate()` and `SwarmWorker.execute()` as
  plain async APIs, with every LLM backend injected as a deterministic fake.
- Failure containment: a worker exception is captured as
  `WorkerResult(success=False, error=...)` — never raised out of the worker — so one
  failure cannot abort the swarm.
- The deterministic re-plan: unmet orders seed round 2 with `attempt + 1` and no LLM
  call (re-planning targets only unmet or failed orders).
- The two LLM-free reduction strategies, `concat` and `first_success`, alongside the
  default `synthesis`.
- Hash-first auditing: a duck-typed audit stand-in prints the `skynet_plan` /
  `skynet_evaluate` records the real `AuditLogger` would append.

## How it works

### A console audit stand-in

`SkynetSupervisor` audits every plan and evaluation. The records are hash-first: the
goal and answer appear only as SHA-256 digests, never as content. The demo swaps in a
tiny in-memory logger that prints each record (dropping the `*_hash` fields for
readability) so you can watch the audit trail as the loop runs:

```python
class _ConsoleAudit:
    """In-memory stand-in for AuditLogger — prints hash-first records to stdout."""

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        compact = {k: v for k, v in payload.items() if not k.endswith("_hash")}
        print(f"  [audit] {event}: {compact}")
```

A `skynet_plan` record carries the round, sizing mode (dynamic vs. fixed), whether
this is a re-plan, and the requested / effective / deferred order counts; a
`skynet_evaluate` record carries completeness, result count, and failure count.

### Fakes with a planted failure

The planner fake emits one order per competitor (dynamic sizing: the planner chooses
N; the supervisor then hard-caps at `min(skynet_max_swarm, parallel_max_workers)`
and defers any overflow onto `plan.deferred`). The worker fake reads the sanitized
user message produced by `SecurePromptBuilder` — `messages[1]` in the
`[system, user]` pair — and fails `ord-2` on its first attempt only:

```python
async def fake_worker(messages: list[dict[str, str]]) -> str:
    user = messages[1]["content"]  # SecurePromptBuilder output: [system, user]
    if "ord-2" in user and "attempt 1" in user:
        raise RuntimeError("upstream rate limited")
    name = next((n for n in _COMPETITORS if n in user), "unknown")
    return f"{name}: strong in EU, weak mobile offering, pricing mid-market."
```

That the fake can key off `"attempt 1"` is itself instructive: the worker's user
prompt is composed as `Order <id> (attempt <n>): <instruction>`, so the attempt
counter is visible to the model on retries. The evaluator fake declares the goal
unmet while any result line reads `FAILED` — mirroring how the real evaluator sees a
status-annotated summary of worker results through the secure channel.

### Round 1: plan, fan out, reduce, evaluate

The supervisor and worker are constructed with their injected collaborators, then the
loop runs by hand. The fan-out that the subgraph would express as one `Send()` per
order is here a plain `asyncio.gather`:

```python
supervisor = SkynetSupervisor(
    plan_fn=fake_plan, evaluate_fn=fake_evaluate, audit=audit, settings=settings
)
worker = SwarmWorker(worker_fn=fake_worker, settings=settings)

plan = await supervisor.plan(GOAL)
print(f"Round {plan.round}: {plan.size} order(s) — {plan.rationale}")
print(f"  deferred beyond the cap: {len(plan.deferred)}")

results = list(await asyncio.gather(*(worker.execute(order) for order in plan.orders)))
```

`ord-2` comes back as `FAILED (upstream rate limited)` while the other two succeed.
The demo then reduces with both LLM-free strategies — `concat` joins successful
outputs keyed by order id; `first_success` returns the earliest success — and
evaluates. Failed results are excluded from the reduction but retained for audit and
re-planning:

```python
concat = await reduce_results(GOAL, results, strategy="concat", settings=settings)
first = await reduce_results(GOAL, results, strategy="first_success", settings=settings)
complete, answer = await supervisor.evaluate(GOAL, results)
```

`evaluate()` returns `(False, "Partial: ...")`, signalling a re-plan. (In the
subgraph, the conditional edge after evaluation loops back to the plan node while
work remains and the round count is below `skynet_max_rounds`; it also enforces the
`skynet_token_budget` at each round boundary via the shared `CostMeter`, raising
`SkynetBudgetExceeded` on breach.)

### Round 2: the deterministic re-plan

Passing `unmet=` to `plan()` short-circuits the planner backend entirely: unmet
orders pass through with `attempt` incremented, and the rationale records that this
round is a re-plan. No LLM call, no re-decomposition of already-satisfied work:

```python
failed_ids = {r.order_id for r in results if not r.success}
unmet = [o for o in plan.orders if o.order_id in failed_ids]
replan = await supervisor.plan(GOAL, round=2, unmet=unmet)
for order in replan.orders:
    print(f"  [{order.order_id}] attempt={order.attempt}: {order.instruction}")

retries = list(await asyncio.gather(*(worker.execute(order) for order in replan.orders)))
```

On attempt 2 the fake worker succeeds. The demo merges round-1 successes with the
retries, reduces once more with `concat`, and evaluates again — this time
`complete=True`, and the evaluator's synthesized text is the final answer. This is
precisely the loop the subgraph automates, minus the graph plumbing.

## Key API

| Symbol | Role |
|---|---|
| `SkynetSupervisor` | Owns sizing and the control loop; `plan(goal, round=, unmet=)` and `evaluate(goal, results)` |
| `SkynetSupervisor.plan()` | Decomposes via `plan_fn` (or passes unmet orders through with `attempt + 1`); caps and defers overflow |
| `SkynetSupervisor.evaluate()` | Returns `(complete, answer)`; audits `skynet_evaluate` hash-first |
| `SwarmWorker.execute()` | Runs one `SwarmOrder`; failures become `WorkerResult(success=False)`, never exceptions |
| `reduce_results()` | Merges successes: `synthesis` (default, LLM) &#124; `concat` &#124; `first_success` (both LLM-free) |
| `SwarmOrder` / `SwarmPlan` / `WorkerResult` | Frozen value objects flowing through the loop |

## Run it

```bash
uv run jupyter lab notebooks/skynet_direct_api.ipynb
uv run python examples/skynet_direct_api.py   # from the prismal repo
```

API key: not required — the demo runs fully offline with injected fakes.

## Related

- [Skynet swarm](skynet_swarm.md) — the same loop wrapped as a LangGraph subgraph with `Send()` fan-out.
- [Skynet specialist swarm](skynet_specialist_swarm.md) — S+ specialist roles, metering, and remote A2A workers.
- [Supervisor quickstart](supervisor_quickstart.md) — the main supervisor graph Skynet plugs into.
- [Observability integration](observability_integration.md) — the OTel spans (`skynet.plan`, `skynet.worker`, ...) emitted along this path.
