# Cost and budget governance with injected fakes (Phase C)

> **Notebook:** [`notebooks/budget_governance.ipynb`](../notebooks/budget_governance.ipynb) · **Example:** [`examples/budget_governance.py`](https://github.com/prismal-ai/prismal/blob/main/examples/budget_governance.py)

Phase C is the **enforcement** layer built atop `prismal/monitoring/`, which only *observes*: meter real per-run usage, compare it against a declared `Budget`, and cut off when the ceiling is reached — softly (patterns degrade) or hard (the run aborts with `BudgetExceeded`). The whole layer is opt-in behind `settings.budget_enabled` (default `False`); with the flag off there is zero extra state and the disabled path is byte-for-byte unchanged.

This walkthrough exercises the engine's core objects — `CostMeter`, `Budget`, `BudgetGuard`, and the `make_budget_guard_fn` adapter — completely offline. Fake LLM responses carry LangChain-style `usage_metadata`, and a pricing table in `Settings` makes the USD cost deterministic, so no API key or network is required.

## What it demonstrates

- Metering fake LLM responses into a `CostMeter` and watching a cumulative `Usage` grow call by call (tokens, USD cost, call count).
- A `BudgetGuard` verdict moving through **within → soft → hard** as spend approaches and then crosses the ceiling; `check()` is a pure O(1) read with no side effects.
- The `Budget` conventions: one ceiling per dimension (`max_tokens`, `max_cost_usd`, `max_calls`, `max_wall_clock_s`), where `0` means *unlimited* on that dimension.
- `make_budget_guard_fn(guard)` adapting a guard into the `budget_guard_fn` callable the expensive patterns consume — here `reflection_loop` stops early on a soft cap instead of looping to `max_iterations`.
- Deterministic offline pricing via `settings.budget_pricing` (the fallback table `compute_cost_usd()` uses when LiteLLM has no price for the model).

## How it works

### A fake response with real usage metadata

The engine never talks to a provider SDK. `CostMeter.record_response()` calls `extract_token_usage()`, which reads the LangChain-standard `usage_metadata` off a message (falling back to the OpenAI-style `response_metadata["token_usage"]`, then to zeros — it never raises). Any object exposing that attribute is meterable, so a `SimpleNamespace` is enough:

```python
def _fake_response(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    """A stand-in for a LangChain AIMessage carrying usage metadata."""
    return SimpleNamespace(
        usage_metadata={"input_tokens": prompt_tokens, "output_tokens": completion_tokens}
    )
```

### Meter, budget, guard

The demo builds `Settings` with a pricing entry for the made-up `demo/model` (per-1k input/output token rates). `compute_cost_usd()` tries LiteLLM's native model map first; since `demo/model` is unknown there, it falls through to this table, and the resulting `Usage` is flagged `estimated=True`. Then a meter, a budget, and a guard are assembled:

```python
settings = Settings(budget_pricing={"demo/model": {"input": 1.0, "output": 2.0}})

meter = CostMeter(settings=settings)
budget = Budget(max_tokens=4_000, max_cost_usd=10.0, max_calls=5)
guard = BudgetGuard(budget, meter, soft_ratio=0.8)
```

`Budget` is a frozen value object scoped to `turn` by default (the other scopes are `session` and `tenant`); dimensions left at `0` are unlimited. `BudgetGuard` compares the meter's cumulative `Usage` against the budget: any dimension at or above `soft_ratio` of its limit is a **soft** breach, at or above the limit itself a **hard** breach.

### Watching the verdict escalate

Each loop iteration meters one 700-token response (`record_response()` does extract → price → record, in-memory and O(1)) and reads the verdict:

```python
for i in range(1, 6):
    usage = meter.record_response(_fake_response(500, 200), "demo/model", agent="demo")
    status = guard.check()
    level = "HARD" if status.hard_exceeded else "soft" if status.soft_exceeded else "ok"
    print(
        f"call {i}: tokens={usage.total_tokens:>5} "
        f"cost=${usage.cost_usd:5.2f} calls={usage.calls} → {level}"
    )
    if status.hard_exceeded:
        break
```

`Usage` is cumulative and summable with `+` (its `estimated` flag is the OR of the operands), so `record_response()` simply adds each call's usage onto the running total. Against the 4,000-token / 5-call budget, calls 1–4 accumulate 700 tokens each: the run is `ok` until the ratio crosses 0.8 (soft), and the fifth call trips a hard breach — `BudgetStatus.breached_dimension` names which ceiling gave way. Note that `check()` itself never raises; raising is the job of `enforce()` (and only when `hard_cap=True`), which the [graph-integration walkthrough](budget_graph_integration.md) covers.

### A pattern degrades on a soft cap

The second half shows the consumption side. A fresh meter is pre-loaded to 85% of a tiny 100-token budget, so the guard is already in soft territory, and `hard_cap=False` keeps `enforce()` from raising:

```python
tight_meter = CostMeter()
tight_meter.record(Usage(prompt_tokens=85))
tight = BudgetGuard(Budget(max_tokens=100), tight_meter, soft_ratio=0.8, hard_cap=False)
guard_fn = make_budget_guard_fn(tight)
```

`make_budget_guard_fn` adapts the guard to the uniform `async fn(ctx) -> bool` contract the expensive patterns accept (`reflection_loop`, `debate_round`, `tree_of_thoughts`, `LATSAgent.search`, `MixtureOfAgents.generate`): `True` means proceed, `False` means degrade/stop, and a hard breach raises `BudgetExceeded` when `hard_cap` is set. Passing `None` instead of a guard yields a zero-overhead always-`True` function — the disabled path, which is why un-budgeted callers pay nothing.

The demo hands `guard_fn` to a reflection loop whose critique never meets the threshold, so it *would* run all five iterations:

```python
best, score = await reflection_loop(
    generate, critique, {}, threshold=0.9, max_iterations=5, budget_guard_fn=guard_fn
)
print(f"reflection: generated {gen_calls['n']} draft(s) before the budget stopped it")
```

Because the guard is already soft, the pattern is stopped after a single draft: the budget, not the iteration cap, ends the loop — the essence of soft-cap degradation.

### Where the numbers come from

Two design decisions make the metering path safe to leave on everywhere. First, extraction and pricing **never raise**: `extract_token_usage()` tries the LangChain `usage_metadata`, then the OpenAI-style `response_metadata["token_usage"]`, and finally yields zeros, while `compute_cost_usd()` degrades to a zero-cost `"none"` estimate when neither LiteLLM nor the pricing table knows the model — a metering failure can never break a turn. Second, `providers/cost.py` is the single new module that imports `litellm`, respecting the framework's rule that provider SDK imports live only in `prismal/providers/`; `budget/usage.py` handles LangChain conventions and therefore stays outside it.

Beyond enforcement, every `record()` also emits OTel counters tagged with `agent`, `pattern`, `model`, and `tenant` for cost attribution, and — when a `CostTracker` plus `session_id` are injected into the meter — bridges each call into persistent FinOps history. Enforcement only ever reads `meter.usage`; it never depends on that persistence bridge, so the guard works identically with or without it.

## Key API

| Symbol | Role |
|---|---|
| `Budget` | Frozen per-dimension spend ceiling (`max_tokens`, `max_cost_usd`, `max_calls`, `max_wall_clock_s`); `0` = unlimited; scoped `turn \| session \| tenant`. |
| `Usage` | Cumulative spend for one run; summable with `+`; `estimated` ORs across operands. |
| `CostMeter` | Per-run accumulator; `record_response()` does extract → price → record; in-memory, no hot-path I/O. |
| `BudgetGuard` | Compares meter usage to the budget; `check()` is a pure O(1) verdict (within / soft / hard). |
| `make_budget_guard_fn(guard)` | Adapts a guard to the `budget_guard_fn` patterns consume; `None` → zero-overhead always-`True`. |
| `reflection_loop(...)` | Generate → critique → refine pattern; honours `budget_guard_fn` before each iteration. |
| `Settings.budget_pricing` | Per-model per-1k fallback pricing table used when LiteLLM's model map has no price. |

## Run it

```bash
uv run jupyter lab notebooks/budget_governance.ipynb
uv run python examples/budget_governance.py   # from the prismal repo
```

No API key is required — the demo runs fully offline with fake responses and a deterministic pricing table. In the notebook, the first cell applies `nest_asyncio`, and the final cell contains a commented `# await main()`: uncomment it and run the cell to execute the end-to-end demo inside Jupyter's event loop.

## Related

- [Budget seeding at the graph seam](budget_graph_integration.md) — how a host wires this engine around a graph run.
- [Skynet swarm](skynet_swarm.md) — the swarm layer builds a `Budget(max_tokens=skynet_token_budget)` over a shared meter.
- [Observability integration](observability_integration.md) — the OTel counters and histograms the meter and guard emit.
- [Runtime hardening](runtime_hardening.md) — the neighbouring loop/limit controls at the node seam.
