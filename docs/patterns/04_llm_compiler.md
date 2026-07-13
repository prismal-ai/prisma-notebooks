# LLM-Compiler (Parallel DAG)

> **Notebook:** [`notebooks/patterns/04_llm_compiler.ipynb`](../../notebooks/patterns/04_llm_compiler.ipynb) · **Example:** [`examples/patterns/04_llm_compiler.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/04_llm_compiler.py)

LLM-Compiler (SPEC-PAT-005) treats a complex goal the way a compiler treats source code: it *plans* the goal into a directed acyclic graph (DAG) of tool-calling tasks, *validates* the graph, *schedules* independent tasks into concurrent waves, and finally *joins* all outputs into one answer. Compared with a sequential ReAct-style loop — one tool call, one observation, repeat — the compiler executes everything that does not depend on anything else simultaneously, cutting wall-clock latency dramatically for multi-source questions.

The notebook grounds the pattern in **HotpotQA**, a dataset of 113k multi-hop questions that require reasoning over multiple documents ("what is the capital of the country where Python was invented, and how many people live there?"). Such questions decompose naturally into a DAG: search A and B independently, then synthesize with both results.

## What it demonstrates

- Decomposing a natural-language goal into a `TaskNode` DAG via a pluggable `plan_fn`.
- DAG validation with **Kahn's algorithm** — cycles, duplicate ids, and unknown dependencies are rejected with `CompilerError` before anything runs.
- Wave-based scheduling: `compute_execution_waves()` groups tasks by topological depth, and each wave runs concurrently with `asyncio.gather`.
- Data flow between tasks with the `$TaskId.output` placeholder, interpolated into downstream `args` just before execution.
- Re-planning on failure: failed tasks trigger a fresh plan (up to `max_replanning` times) that receives the accumulated results as context.

## How it works

### Injected callables instead of hard-wired LLM calls

Like every prismal pattern, `LLMCompiler` is decoupled from any specific model: it accepts three async callables — `plan_fn` (goal → task list), `tool_executor` (task + prior outputs → output), and `joiner` (goal + completed tasks → answer). The notebook keeps them deterministic (keyword heuristics and mock tool results) so the DAG mechanics are observable without API latency; in production `plan_fn` would be an LLM planner and `tool_executor` would hit real search/scraping tools.

### Planning a question into a DAG

`plan_fn` returns a list of `TaskNode` objects whose `depends_on` edges define the graph. For the Python-capital question, the plan is a linear chain; for the framework-comparison question it fans out three independent lookups that converge on a comparison task:

```python
tasks = [
    TaskNode(id="T1", description="Gather LangChain data from GitHub",
             tool="web_scrape", args={"url": "..."}, depends_on=[]),
    TaskNode(id="T2", description="Gather LlamaIndex data from GitHub",
             tool="web_scrape", args={"url": "..."}, depends_on=[]),
    TaskNode(id="T3", description="Find enterprise adoption of both frameworks",
             tool="search", args={"query": "..."}, depends_on=[]),
    TaskNode(id="T4", description="Compare the three results", tool="compare",
             args={"langchain_data": "$T1.output",
                   "llamaindex_data": "$T2.output",
                   "enterprise_data": "$T3.output"},
             depends_on=["T1", "T2", "T3"]),
]
```

The `"$T1.output"` strings are the compiler's data-flow mechanism: right before a task runs, every placeholder in its `args` is replaced with the real output of the referenced task. The original `args` are restored afterwards, so the final plan remains a faithful record of what was planned.

The `plan_fn` signature also receives `previous_results` — `None` on the first attempt, and the partial output map when re-planning. The notebook uses it to inject a recovery task (`T_verify`) whenever a prior plan left failures behind.

### Executing waves concurrently

`compile_and_run()` drives the full loop: plan → validate → execute → join. Execution walks the topological waves; every task in a wave is awaited together:

```python
compiler = LLMCompiler(
    tools=[{"name": t} for t in _MOCK_TOOL_RESULTS],
    plan_fn=plan_fn,
    tool_executor=tool_executor,
    joiner=joiner,
    max_replanning=2,
)
result = await compiler.compile_and_run(
    goal=sample["question"],
    state={"messages": [], "metadata": {}},
)
```

For the fan-out plan above the compiler prints two waves — `["T1", "T2", "T3"]` executed simultaneously, then `["T4"]` — instead of four sequential steps. `tool_executor` failures don't abort the wave (`asyncio.gather(..., return_exceptions=True)`); failed tasks are marked `status="failed"` and, if any exist, the compiler re-plans with the partial results. Exceeding `max_replanning` raises `CompilerError`.

The notebook's executor stands in for real tools with a small latency simulation, but its signature is the production one — it receives the task (with placeholders already interpolated) and a map of prior outputs keyed by task id:

```python
async def tool_executor(task: TaskNode, prior_outputs: dict[str, Any]) -> str:
    await asyncio.sleep(0.05)  # simulate tool latency
    tool_result = _MOCK_TOOL_RESULTS.get(task.tool, "result not available")
    if prior_outputs:
        context_ids = ", ".join(prior_outputs.keys())
        return f"[{task.tool}] {task.description} → {tool_result} (using context: {context_ids})"
    return f"[{task.tool}] {task.description} → {tool_result}"
```

Once every task completes, the `joiner` receives the goal and the full list of completed `TaskNode`s (outputs attached) and synthesizes the final answer — in production this is typically one last LLM call over all gathered evidence.

### Validation and result inspection

The notebook finishes by validating a diamond-shaped DAG explicitly:

```python
plan = CompilerPlan(tasks=valid_plan_tasks, goal="test", ...)
is_valid = compiler.validate_dag(plan)          # True, or raises CompilerError
waves = compiler.compute_execution_waves(valid_plan_tasks)
# A → B,C → D  ⇒  waves == [["A"], ["B", "C"], ["D"]]
```

The returned `CompilerResult` carries the `final_answer` from the joiner, the executed `plan` (with per-task outputs and statuses), plus `replanning_count`, `tasks_succeeded`, and `tasks_failed` for observability. The whole run is wrapped in an OpenTelemetry span (`llm_compiler.run`).

## Key API

| Symbol | Role |
|---|---|
| `LLMCompiler` | Orchestrator: plan → Kahn validation → wave execution → join, with bounded re-planning (`max_replanning`, default 2). |
| `TaskNode` | One DAG node: `id`, `description`, `tool`, `args`, `depends_on`, plus runtime `output` and `status` (`pending/running/completed/failed`). |
| `CompilerPlan` | Validated plan: `tasks`, precomputed `execution_waves`, `goal`; `to_json()` for debugging. |
| `CompilerResult` | Outcome of `compile_and_run()`: `final_answer`, executed `plan`, `replanning_count`, `tasks_succeeded/failed`. |
| `LLMCompiler.validate_dag()` | Kahn's algorithm check; raises `CompilerError` on cycles, duplicate ids, or unknown deps. |
| `LLMCompiler.compute_execution_waves()` | Groups tasks into topological waves safe to run concurrently. |
| `PlanFn` / `ToolExecutorFn` / `JoinerFn` | Type aliases for the three injected async callables. |

## Run it

```bash
uv run jupyter lab notebooks/patterns/04_llm_compiler.ipynb
# or the script version:
uv run python examples/patterns/04_llm_compiler.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — though the notebook's mock callables let the DAG mechanics run deterministically.

## Related

- [Parallel Dispatcher](09_parallel_dispatcher.md) — the LangGraph `Send()` primitive for fan-out; LLM-Compiler adds planning, dependencies, and re-planning on top of raw parallelism.
- [Skynet swarm](../skynet_swarm.md) — swarm map-reduce over agents: plan → fan-out workers → reduce, the agent-level cousin of the compiler's task DAG.
- [Tree of Thoughts](01_tree_of_thoughts.md) — a different decomposition strategy: exploring alternative reasoning branches rather than executing a fixed task graph.
- [Budget governance](../budget_governance.md) — capping tokens/cost when parallel patterns multiply LLM calls.
