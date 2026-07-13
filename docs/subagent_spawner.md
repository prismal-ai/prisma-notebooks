# SubAgent Spawner: bounded fan-out with concurrency, timeouts, and cancellation

> **Notebook:** [`notebooks/subagent_spawner.ipynb`](../notebooks/subagent_spawner.ipynb) · **Example:** [`examples/subagent_spawner.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subagent_spawner.py)

`SubAgentSpawner` (`prismal/agents/spawner.py`) is prismal's low-level launcher for dynamic
sub-agents: it runs any coroutine as an independent `asyncio.Task` while enforcing a global
concurrency limit, a per-task timeout, and an emergency `cancel_all()` switch. Inspired by the
picoClaw `spawn` tool concept, it is the raw machinery underneath prismal's higher-level fan-out
features — where `parallel_research` and the Skynet swarm decide *what* to run in parallel, the
spawner governs *how many* things run at once and *for how long*.

Reach for it when you need controlled parallelism without a full LangGraph subgraph: batch
delegation, background workers, or any place where an unbounded `asyncio.gather()` would risk
resource exhaustion. This notebook exercises all three control features with deterministic fake
worker coroutines — no LLM, no network, no API keys — so the scheduling behavior itself is what
you observe.

## What it demonstrates

- Spawning four workers under a two-slot semaphore (`max_concurrent_agents=2`), so the third and
  fourth only launch once an earlier worker releases a slot.
- Per-task timeout enforcement: a worker that overruns its deadline is cancelled and surfaces a
  `TimeoutError` to whoever awaits it.
- `cancel_all()` as a kill switch that force-cancels every still-running task and reports how many
  it stopped.
- Driving the spawner from an explicit `Settings` instance instead of the global cached
  `get_settings()`, keeping the demo self-contained and reproducible.
- `active_count` and `cleanup()` for observing and pruning the internal task registry.

## How it works

### A deterministic stand-in for a sub-agent

The demo never calls a model. A fake worker sleeps for a fixed duration and returns a report
string, which makes the semaphore and timeout mechanics fully deterministic:

```python
async def fake_worker(name: str, duration: float) -> str:
    """Deterministic stand-in for a sub-agent run (sleeps, then reports)."""
    await asyncio.sleep(duration)
    return f"{name}: analysed its shard in {duration:.2f}s"
```

In a real deployment the coroutine would be an agent run — for example a graph invocation or a
specialist node — but the spawner is agnostic: it accepts any zero-argument
`coro_factory` and returns the launched `asyncio.Task`.

### Fan-out under a two-slot semaphore

The spawner is constructed from an explicit `Settings` instance. `max_concurrent_agents` sizes an
internal `asyncio.Semaphore`; `agent_timeout_seconds` becomes the default per-task timeout.

```python
settings = Settings(max_concurrent_agents=2, agent_timeout_seconds=300)
spawner = SubAgentSpawner(settings=settings)
```

The first demo spawns four workers. The crucial detail is that `spawn()` *awaits the semaphore
before launching*: it is backpressure at the call site, not a queue. With two slots, the calls for
workers 3 and 4 block until an earlier worker finishes and frees a slot.

```python
tasks = []
for i in range(1, 5):
    # spawn() awaits the semaphore: workers 3 and 4 only launch once an
    # earlier worker frees a slot.
    task = await spawner.spawn(
        coro_factory=lambda i=i: fake_worker(f"worker-{i}", 0.05 * i),
        task_id=f"worker-{i}",
    )
    tasks.append(task)
    print(f"  spawned worker-{i} (active={spawner.active_count})")

results = await asyncio.gather(*tasks)
```

Internally each task runs inside a wrapper coroutine that applies `asyncio.wait_for()`, logs
timeout/cancellation/error outcomes through the structured logger, and — in a `finally` block —
releases the semaphore and removes the task from the registry. The slot is therefore returned no
matter how the task ends, which is what makes the fan-out loop safe.

### Timeout enforcement

Each `spawn()` accepts an optional `timeout` that overrides the default. The second demo gives a
5-second worker a 0.1-second budget:

```python
task = await spawner.spawn(
    coro_factory=lambda: fake_worker("slow-worker", 5.0),
    task_id="slow-worker",
    timeout=0.1,
)
try:
    await task
except TimeoutError:
    print("  slow-worker exceeded its 0.1s timeout and was cancelled")
```

The `TimeoutError` propagates to the awaiter rather than being swallowed, so callers decide
whether a late worker is fatal, retryable, or ignorable. This is the same discipline the framework
applies elsewhere (for example, the `@prismal_node` middleware wraps user functions in
`asyncio.wait_for`): a sub-agent never gets to run unbounded.

### `cancel_all()` as the kill switch

The last demo starts two 30-second workers, then cancels everything in flight:

```python
tasks = [
    await spawner.spawn(
        coro_factory=lambda i=i: fake_worker(f"long-{i}", 30.0),
        task_id=f"long-{i}",
        timeout=60.0,
    )
    for i in (1, 2)
]
cancelled = spawner.cancel_all()

outcomes = await asyncio.gather(*tasks, return_exceptions=True)
```

`cancel_all()` returns the number of tasks that were actually cancelled (already-finished tasks
are skipped), and gathering with `return_exceptions=True` shows each task resolving to
`CancelledError`. Finally, `spawner.cleanup()` prunes completed entries from the registry and
`active_count` drops back to zero. Every spawn, timeout, and forced cancellation is emitted
through prismal's structured logger (`spawning_sub_agent`, `sub_agent_timeout`,
`sub_agent_force_cancelled`), so real runs leave an observable trail.

## Key API

| Symbol | Role |
|---|---|
| `SubAgentSpawner(settings=None)` | Async sub-agent launcher; reads `max_concurrent_agents` and `agent_timeout_seconds` from `Settings` (or the global cached settings). |
| `spawner.spawn(coro_factory, task_id, timeout=None)` | Awaits the semaphore, then launches the coroutine as a named `asyncio.Task` wrapped in `asyncio.wait_for`; returns the task. |
| `spawner.cancel_all()` | Force-cancels every non-done task; returns the count cancelled. |
| `spawner.cleanup()` | Removes completed tasks from the internal registry. |
| `spawner.active_count` | Number of currently active (non-done) tasks. |
| `Settings(max_concurrent_agents=..., agent_timeout_seconds=...)` | Explicit configuration used by the demo instead of the cached global settings. |

## Run it

```bash
uv run jupyter lab notebooks/subagent_spawner.ipynb
uv run python examples/subagent_spawner.py   # from the prismal repo
```

No API key required — the demo runs fully offline with fake worker coroutines.

## Related

- [Parallel research fan-out](parallel_research_fanout.md) — the LangGraph `Send()`-based fan-out pattern one level up.
- [Skynet swarm](skynet_swarm.md) — plan → fan-out → reduce with the same concurrency caps applied at swarm scale.
- [Supervisor quickstart](supervisor_quickstart.md) — the state machine that dynamic sub-agents plug into.
- [Budget governance](budget_governance.md) — complementary limits on tokens and cost rather than concurrency and time.
