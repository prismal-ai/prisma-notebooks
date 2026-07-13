# Agent Evaluation Harness: Assertions, Scorecards, and Regression Gates

> **Notebook:** [`notebooks/agent_eval.ipynb`](../notebooks/agent_eval.ipynb) · **Example:** [`examples/agent_eval.py`](https://github.com/prismal-ai/prismal/blob/main/examples/agent_eval.py)

Guardrails, budgets, and hardening protect a run while it happens; the Phase V eval harness answers the question that comes *before* shipping: does the agent still behave? `prismal.eval` runs declarative `EvalSet`s of cases through the compiled supervisor graph, captures a structural `Trajectory` of each run (final answer, visited nodes, tool calls, tool errors, tokens, cost, latency), checks it against typed assertions — exact match, semantic similarity, LLM-judge rubrics, tool-usage bounds, groundedness, and security containment — and aggregates everything into a `Scorecard` that a regression gate can compare against a committed baseline in CI.

Like everything else in the framework, the runner is factory-injected: `EvalRunner(graph_factory=..., runtime_factory=...)` accepts callables for the graph and the runtime, so this notebook substitutes a scripted graph double and a no-op runtime and executes the *entire* harness — trajectory capture, assertion dispatch, scorecard aggregation, markdown/JSON reports, regression comparison — offline with no API keys. Drop the arguments (`EvalRunner()`) and the same code runs against the real `get_async_compiled_graph()` with `build_test_runtime` fakes.

## What it demonstrates

- Building an `EvalRunner` around injected `graph_factory` / `runtime_factory` callables — the same seam the defaults fill with the real compiled graph and composition-root fakes.
- Declaring an `EvalSet` with an `EXACT`-match case and a `TOOL_USAGE` case bounding the number of steps.
- The trajectory-capture pass rule: a case passes only if the run **terminated** cleanly *and* every assertion held.
- Rendering the resulting `Scorecard` as Markdown and JSON reports.
- Gating with `compare()` — the regression check CI runs against a committed baseline scorecard.

## How it works

### Injected fakes: a scripted graph and a no-op runtime

The harness talks to the graph exclusively through `astream(..., stream_mode="updates")` — the same interface the real compiled supervisor exposes. So a test double only needs to replay update dicts. The runtime double only needs a `tool_provider` attribute and an `aclose()`:

```python
class _ScriptedGraph:
    """A graph double whose astream replays a fixed answer for any input."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def astream(self, _input, _config=None, *, stream_mode="updates"):
        yield {"supervisor": {"messages": [AIMessage(content=self._answer)]}}


class _NoopRuntime:
    tool_provider = None

    async def aclose(self) -> None:
        return None
```

`_runner(answer)` wraps both in factories and hands them to `EvalRunner(graph_factory=..., runtime_factory=...)`. When these arguments are omitted, the defaults bind lazily to `get_async_compiled_graph` (with the runtime's tool provider injected) and `composition.build_test_runtime` — deterministic fakes, no I/O.

### Declaring cases and assertions

Eval content is data, not code: frozen `EvalCase`s each carry an input and a list of typed `Assertion`s (sets can also load from YAML via `EvalSet.from_yaml`). Six `AssertionType`s exist — `EXACT`, `SEMANTIC` (embedding cosine ≥ `min_score`, default 0.8), `LLM_JUDGE` (rubric scored by a `Judge`, default threshold 0.7), `TOOL_USAGE`, `GROUNDEDNESS`, and `SECURITY` (attack containment). The demo uses the two that need no model at all:

```python
eval_set = EvalSet(
    suite="example",
    cases=[
        EvalCase(
            id="greeting",
            input="Say hi.",
            assertions=[Assertion(type=AssertionType.EXACT, expected="hi there")],
        ),
        EvalCase(
            id="tool-bounds",
            input="Answer directly.",
            assertions=[Assertion(type=AssertionType.TOOL_USAGE, max_steps=3)],
        ),
    ],
)
```

`EXACT` is a whitespace-trimmed string comparison against the final answer. `TOOL_USAGE` checks structure instead of text: every tool in `must_call` was used, nothing in `never_call` was, and the trajectory took at most `max_steps` steps. Assertion checkers are pure functions that never raise — a failure is a `passed=False` result with a human-readable `detail`.

### Running the set: trajectory capture and the pass rule

`run_set()` executes cases sequentially through `run_case()`, which never raises: it seeds the RNG (per-case `setup["seed"]` or `settings.eval_seed`), builds the runtime and graph, and drives the stream with `thread_id=f"eval-{case.id}"`. Trajectory capture folds every update into a `Trajectory`: the **final answer** is the last assistant message that requested no tool calls; each tool request, tool result, and text message becomes a `TrajectoryStep`; token usage and USD cost come from the same `extract_token_usage` / `compute_cost_usd` machinery the budget layer uses. The pass rule is deliberately strict:

```python
card = await _runner("hi there").run_set(eval_set)
# passed = trajectory.terminated and all(ar.passed for ar in assertion_results)
```

A run that crashed mid-stream (`terminated=False`) fails its case even if every assertion would have held — an eval harness that shrugs off exceptions would hide exactly the regressions it exists to catch. The aggregated `Scorecard` carries `pass_rate`, `avg_steps`, `tool_error_rate`, `avg_cost_usd`, `avg_latency_ms`, and the per-case results.

### Reports and the regression gate

`to_markdown(card)` renders a human-readable scorecard (headline metrics plus a ✅/❌ per-case table) and `to_json(card)` a stable, sorted JSON document suitable for committing as a baseline. `compare()` is the CI gate:

```python
print(to_markdown(card))
print(to_json(card))

result = compare(card, card)   # current vs. baseline
print(f"regression gate passed: {result.passed}")
```

The gate is tolerance-based (default 2%): `pass_rate` may not drop by more than the tolerance, while `avg_steps`, `tool_error_rate`, and `avg_cost_usd` may not *rise* beyond it (relative when the baseline is non-zero). Latency is deliberately not gated — too machine-dependent. Each violated metric lands in `result.regressions`, so the failure message names exactly what got worse. Comparing a card against itself, as here, trivially passes.

Scores also flow onward: `to_langfuse()` posts the suite's pass rate as trace feedback when `eval_langfuse_export` is enabled, and LLM-judge scores plug into the observability port's `record_score()` — so eval results, production traces, and dashboards all key off the same run-naming convention.

## Key API

| Symbol | Role |
|---|---|
| `EvalRunner` | Orchestrator; injectable `graph_factory` / `runtime_factory` / `evaluator` / `judge`, defaults bound lazily to the real graph + `build_test_runtime` |
| `EvalSet` / `EvalCase` / `Assertion` | Frozen declarative eval content; `EvalSet.from_yaml()` loads suites from disk |
| `AssertionType` | `EXACT` · `SEMANTIC` · `LLM_JUDGE` · `TOOL_USAGE` · `GROUNDEDNESS` · `SECURITY` |
| `Trajectory` / `TrajectoryStep` | Structural capture of one run: final answer, steps, visited nodes, tool calls/errors, tokens, cost, `terminated` |
| `Scorecard` / `CaseResult` / `AssertionResult` | Aggregated suite metrics and per-case / per-assertion outcomes |
| `compare()` / `RegressionResult` | Tolerance-based baseline gate (`passed`, named `regressions`); latency excluded |
| `to_markdown()` / `to_json()` | Human-readable and machine-stable scorecard reports |
| `Settings.eval_*` | `eval_default_mode` (`fakes`/`live_api`), `eval_judge_model`, `eval_regression_tolerance`, `eval_seed`, `eval_langfuse_export` |

## Run it

```bash
uv run jupyter lab notebooks/agent_eval.ipynb
uv run python examples/agent_eval.py   # from the prismal repo
```

No API key is required — the scripted graph and no-op runtime keep everything offline (only `SEMANTIC` / `LLM_JUDGE` / `GROUNDEDNESS` assertions would need a model). The notebook's final run cell ships commented out as `# await main()`; the `nest_asyncio.apply()` in the first cell lets the coroutine run directly on Jupyter's event loop.

## Related

- [Observability integration](observability_integration.md) — where LLM-judge scores land via `record_score()` and runs export as eval datasets
- [Guardrails modernization](guardrails_modernization.md) — the controls a `SECURITY` assertion verifies stayed effective
- [Composition root](composition_root.md) — `build_test_runtime`, the deterministic runtime the default `runtime_factory` uses
- [Supervisor quickstart](supervisor_quickstart.md) — the real compiled graph the default `graph_factory` evaluates
