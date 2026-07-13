# Budget seeding at the graph seam — seed, guard, degrade, abort, clear (Phase C)

> **Notebook:** [`notebooks/budget_graph_integration.ipynb`](../notebooks/budget_graph_integration.ipynb) · **Example:** [`examples/budget_graph_integration.py`](https://github.com/prismal-ai/prismal/blob/main/examples/budget_graph_integration.py)

Phase C's enforcement engine has to meet one hard constraint when it touches the LangGraph runtime: a live `CostMeter` (which may hold a SQLite-backed `CostTracker`) must never reach the checkpoint serializer. The design that follows is a small inversion — the per-run `{meter, guard}` pair lives in an **in-process registry keyed by `session_id`**, while checkpointed `AgentState` carries only a serializable marker, `state["metadata"]["budget"] = {"enabled": True, "scope": ...}`. It is the same principle the multimodal layer uses for path-based media descriptors: heavy live objects stay out of state; state carries a pointer-shaped fact.

This walkthrough plays the host's role end to end, fully offline: build `Settings` with `budget_enabled=True` and deliberately small caps, seed the engine for one run, retrieve the `BudgetGuard` at the node seam, meter fake responses until the soft and then the hard threshold fire, watch `BudgetExceeded` raised at the seam, and finally release the engine.

## What it demonstrates

- `resolve_budget(settings)` turning flat budget settings into a frozen `Budget` value object (with its `turn | session | tenant` scope).
- `seed_budget_run(state, settings)` installing the per-run engine in the in-process registry and writing only the checkpoint-safe marker into state — a no-op when `budget_enabled` is `False`.
- `get_budget_guard(state)` recovering the live guard at the node seam via the state's `session_id`.
- The full escalation: metered calls go **ok → soft (degradation advice) → hard**, and the seam-side `guard_fn` raises `BudgetExceeded` because `hard_cap` defaults to `True`.
- `clear_budget_run(state)` releasing the engine when the run finishes (idempotent).

## How it works

### Enabling the budget in settings

Everything is driven by settings, so a host enables governance without touching graph code. The caps here are tiny on purpose — three 400-token calls will walk a 1,000-token ceiling through every stage — and the pricing table keeps the offline USD figure deterministic:

```python
settings = Settings(
    budget_enabled=True,
    budget_max_tokens=1_000,
    budget_max_calls=10,
    budget_soft_ratio=0.8,
    budget_pricing={"demo/model": {"input": 1.0, "output": 2.0}},
)
print("Resolved budget:", resolve_budget(settings))
```

`resolve_budget` maps `budget_max_*` and `budget_scope` onto a `Budget`; dimensions left at `0` (here cost and wall clock) are unlimited. It also accepts an `org_id` hook: with Phase R's `apply_org_overrides`, the settings arriving here are already tenant-resolved, so per-tenant ceilings simply follow.

### Seeding the per-run engine

The host seeds at the graph entry seam, once per run. `seed_budget_run` builds a `CostMeter` and a `BudgetGuard` (with `soft_ratio` and `hard_cap` from settings) and stores them in the registry under the state's `session_id`:

```python
state = create_initial_state(session_id="sess-budget-demo")
seed_budget_run(state, settings)
# Only a serializable marker reaches state; the live engine stays in-process.
print("Checkpoint-safe marker:", state["metadata"]["budget"])
```

The marker is all a checkpoint ever sees: `{"enabled": True, "scope": "turn"}`. When `budget_enabled` is `False`, `seed_budget_run` returns immediately — the disabled path adds zero state and leaves the compiled graph byte-for-byte unchanged. In production the graph seam uses the sibling `maybe_seed_budget_run`, which is idempotent per user-turn signature: it survives the supervisor's many hops within one turn (usage accumulates) and re-seeds a fresh meter when a new user message arrives (turn-scope default).

### Recovering the guard at the node seam

Any node holding the state can recover the live guard through the registry — no engine object travels through the graph:

```python
guard = get_budget_guard(state)
guard_fn = make_budget_guard_fn(guard)  # what the expensive patterns consume
meter = guard.meter
```

`get_budget_guard` returns `None` when the budget is disabled or unseeded, and `make_budget_guard_fn(None)` degrades to a zero-overhead always-`True` function — so node code can wire the adapter unconditionally.

### Driving the run soft, then hard

Each fake response contributes 400 tokens against the 1,000-token ceiling. Call 1 sits at 40% (`ok`); call 2 reaches 80%, exactly the `budget_soft_ratio` (soft); call 3 crosses 100% (hard):

```python
for call in range(1, 4):
    usage = meter.record_response(_fake_response(300, 100), "demo/model", agent="coder")
    status = guard.check()
    level = "HARD" if status.hard_exceeded else "soft" if status.soft_exceeded else "ok"
    if status.soft_exceeded or status.hard_exceeded:
        print(f"    degradation advice: {guard.degradation()}")
    if status.hard_exceeded:
        break
```

`check()` is a pure O(1) verdict with no side effects, so nodes can poll it freely between LLM calls. `degradation()` translates the verdict into pattern advice: `Degradation(reduce=True)` on a soft breach (shrink beams, cut rounds, skip refinement) and `Degradation(terminate=True)` on a hard one. Soft is the *degrade* contract; nothing raises yet.

### The hard cutoff at the seam

Before the next expensive step, the node seam consults `guard_fn`. Under the hood this calls `guard.enforce()`, which — unlike `check()` — has side effects: it writes a hash-first audit record and OTel cutoff counter, and, because `budget_hard_cap` defaults to `True`, raises on a hard breach:

```python
try:
    await guard_fn({"agent": "coder"})
except BudgetExceeded as exc:
    print(
        f"BudgetExceeded raised at the seam: dimension={exc.dimension} "
        f"used={exc.used:.0f} limit={exc.limit:.0f} scope={exc.scope}"
    )
```

The exception carries the breached dimension, the used and limit values, and the scope — enough for a host to return a partial answer with a clear reason. Setting `budget_hard_cap=False` turns a hard breach into log-and-degrade instead of abort.

### Releasing the engine

When the run finishes (or the turn ends), the host releases the registry slot:

```python
clear_budget_run(state)
print("After clear_budget_run:", get_budget_guard(state))  # None
```

The call is idempotent, and afterwards `get_budget_guard` returns `None` — the same shape as the never-seeded path, so downstream code needs no special casing.

## Key API

| Symbol | Role |
|---|---|
| `resolve_budget(settings, *, org_id=None)` | Build a frozen `Budget` from `budget_max_*` settings; per-tenant via already-resolved settings. |
| `seed_budget_run(state, settings)` | Install the per-run `{meter, guard}` in the in-process registry (keyed by `session_id`); write only the serializable marker to state; no-op when disabled. |
| `maybe_seed_budget_run(state, settings)` | Idempotent per-turn variant used at the graph seam; re-seeds only when the user-turn signature changes. |
| `get_budget_guard(state)` | Recover the live `BudgetGuard` for this run, or `None` when disabled/unseeded. |
| `make_budget_guard_fn(guard)` | Adapt the guard to the async `fn(ctx) -> bool` contract patterns and node seams consume. |
| `BudgetGuard.check()` / `.enforce()` / `.degradation()` | Pure verdict / audited cutoff (raises on hard when `hard_cap`) / pattern advice. |
| `BudgetExceeded` | Raised on a hard cap with `hard_cap=True`; carries `dimension`, `used`, `limit`, `scope`. |
| `clear_budget_run(state)` | Release the per-run engine (idempotent). |

## Run it

```bash
uv run jupyter lab notebooks/budget_graph_integration.ipynb
uv run python examples/budget_graph_integration.py   # from the prismal repo
```

No API key is required — the run is fully offline with fake responses and a deterministic pricing table. In the notebook, `nest_asyncio.apply()` in the first cell prepares Jupyter's event loop, and the final cell holds a commented `# await main()`; uncomment it and run the cell to execute the whole seed → degrade → abort → clear sequence.

## Related

- [Cost and budget governance](budget_governance.md) — the engine's building blocks (`CostMeter`, `BudgetGuard`, pattern degradation) in isolation.
- [Composition root](composition_root.md) — where a host assembles ports per tenant; the `org_id` hook feeds `resolve_budget`.
- [Loop hardening](loop_hardening.md) — the complementary iteration/recursion limits at the same node seam.
- [Supervisor quickstart](supervisor_quickstart.md) — the graph whose entry seam `maybe_seed_budget_run` targets.
