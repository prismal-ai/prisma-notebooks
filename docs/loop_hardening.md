# Loop Hardening: Context Compaction and Phase-Scoped Tool Gating

> **Notebook:** [`notebooks/loop_hardening.ipynb`](../notebooks/loop_hardening.ipynb) · **Example:** [`examples/loop_hardening.py`](https://github.com/prismal-ai/prismal/blob/main/examples/loop_hardening.py)

Long-running agent loops degrade in two predictable ways: the message history grows until every LLM call drags hundreds of stale turns along, and the tool catalogue stays maximally wide even when the agent is deep in execution and should no longer be, say, re-planning or browsing. Phase LH hardens the supervisor loop against both. **LH1** is the `ContextCompactor`, which detects an oversized history and shrinks it to a recent verbatim tail (optionally replacing the older segment with an LLM-written summary). **LH2** is phase-scoped tool gating: `resolve_phase()` derives a deterministic loop phase (`planning` / `executing` / `finishing`) from the task-plan fields already in `AgentState`, and `CompositeToolProvider` uses a per-phase capability map to *narrow* — never widen — the tools each agent sees.

Both controls fit prismal's defense-in-depth posture: compaction bounds cost and context-window pressure (complementing the budget layer's token ceilings), and phase gating is least-privilege applied over time — an agent in the executing phase simply cannot reach planning-only tools, shrinking the blast radius of a hijacked loop. The demo runs fully offline: the default `truncate` strategy needs no LLM, and the tool-provider side uses `FakeToolProvider`.

## What it demonstrates

- `should_compact()` flagging a 200-message history against a 60-message ceiling, with a machine-readable reason.
- `compact()` dropping the older segment while keeping a verbatim recent tail, and `to_state_update()` expressing the drop as LangGraph `RemoveMessage` entries.
- `resolve_phase()` moving from `"planning"` to `"executing"` purely from `task_plan` / `pending_tasks` / `completed_tasks` — no LLM call.
- `CompositeToolProvider(phase_capability_map=...)` intersecting an agent's requested capabilities with the phase's allowlist, shrinking the catalogue between a planning call and an executing call.
- The opt-in wiring: both controls default off, with in-process per-run engines and a byte-identical disabled path.

## How it works

### Deciding when to compact

`ContextCompactor` is constructed from `Settings`; `should_compact()` never raises and returns a `(bool, reason)` pair. The primary trigger is message count — strictly greater than `context_compaction_max_messages` — with an optional second trigger (`"token_threshold"`) when a `BudgetGuard` is injected and metered tokens exceed `context_compaction_token_threshold`:

```python
settings = Settings(context_compaction_max_messages=60, context_compaction_keep_recent=10)
compactor = ContextCompactor(settings=settings)

history = [HumanMessage(content=f"turn {i}", id=f"id{i}") for i in range(200)]
should, reason = compactor.should_compact(history)
# -> (True, 'message_count')
```

### Compacting to a verbatim tail

`compact()` splits the history at `keep_recent`: the last N messages survive verbatim; the older segment is dropped (`truncate`, the default — no LLM involved) or replaced by a single summary `AIMessage` (`summarize`, which calls an injectable `summarizer_fn` or an LLM via `ProviderRegistry`, and falls back to truncate on any summarizer failure). The result is a frozen `CompactionResult`:

```python
result = await compactor.compact(history)
# -> compacted=True strategy=truncate messages_dropped=190

update = compactor.to_state_update(result)
# -> {"messages": [RemoveMessage(id="id0"), ... 190 entries]}
```

The state update is the interesting part: compaction never mutates history in place. It emits one `RemoveMessage(id=...)` per dropped message, which LangGraph's `add_messages` reducer interprets as a deletion — so the checkpointed history shrinks through the normal reducer channel, replayable and audit-friendly. Two practical caveats follow from working *with* the reducer rather than around it. First, only messages with stable ids can be removed this way; for ephemeral lists without ids (such as `react_loop`'s local accumulator) the compactor offers `compact_list()`, a position-based variant that rebuilds the list directly. Second, in `summarize` mode the reducer can only append, not splice, so the summary message lands *after* the kept-recent tail — a documented deviation, harmless for LLM consumption.

In the live graph, the supervisor node calls `maybe_compact_context()` each hop, rate-limited by a `context_compaction_min_interval_messages` watermark (default 20 new messages) so the check does not re-run on every turn. The compactor itself is seeded per run into an in-process registry keyed by `session_id` — the same never-checkpoint-live-objects convention the budget and hardening layers use — and the whole feature is opt-in (`context_compaction_enabled`, default `False`): disabled, the loop is byte-for-byte unchanged.

### Resolving the loop phase from plan state

`resolve_phase()` is a pure function over `AgentState` — deterministic, LLM-free, and `None`-safe. An explicit hint at `state["metadata"]["loop"]["phase"]` wins; otherwise the plan fields decide: no `task_plan` means no gating (`None`), a plan with pending tasks is `"planning"` until the first task completes, then `"executing"`, and an exhausted `pending_tasks` list means `"finishing"`:

```python
state = create_initial_state(session_id="demo")
state["task_plan"] = ["scaffold", "implement", "test"]
state["pending_tasks"] = ["scaffold", "implement", "test"]
resolve_phase(state)                      # -> 'planning'

state["completed_tasks"] = ["scaffold"]
state["pending_tasks"] = ["implement", "test"]
resolve_phase(state)                      # -> 'executing'
```

### Narrowing the tool catalogue per phase

`CompositeToolProvider` (the Fase Y merger of MCP, skills, and stub tools) accepts a `phase_capability_map` shaped `{agent: {phase: [capabilities]}}`. When `get_tools()` is called with a `phase`, the agent's requested capabilities are **intersected** with the phase's allowlist — the override can only remove capabilities, never add them, and a phase with no entry applies no narrowing at all:

```python
provider = FakeToolProvider({"coder": []}, default=[])
composite = CompositeToolProvider(
    [provider], phase_capability_map={"coder": {"planning": ["general"]}}
)
composite.get_tools(
    agent_name="coder", capabilities=["general", "code_execution"], phase="planning"
)   # effective capabilities: ['general']  — code_execution gated off while planning
composite.get_tools(
    agent_name="coder", capabilities=["general", "code_execution"], phase="executing"
)   # no map entry for 'executing' -> full capabilities pass through
```

Every actual narrowing increments the `tool_gate_narrowed` OTel counter, and sub-providers that predate the `phase` kwarg are called phase-less (fail-open on `TypeError`) so third-party providers keep working.

In production the map is declared, not coded — `load_phase_capability_map()` reads `config/tool_gating_phases.yaml`:

```yaml
coder:
  planning: [general]            # no code execution while still planning
  executing: [general, code_execution]
researcher:
  finishing: [general]           # wrap-up phase: no more web fan-out
```

A missing file means no gating (the pre-LH2 behavior), while a malformed or non-mapping file raises `ToolGatingConfigError` at startup — configuration errors surface immediately rather than as silently ungated agents.

## Key API

| Symbol | Role |
|---|---|
| `ContextCompactor` | LH1 engine: `should_compact()` → `compact()` → `to_state_update()`; injectable `summarizer_fn` and `budget_guard` |
| `CompactionResult` / `CompactionStrategy` | Frozen outcome (`compacted`, `strategy`, `messages_dropped`, `removed_ids`) and the `truncate` \| `summarize` enum |
| `RemoveMessage` (LangGraph) | The reducer-level deletion marker `to_state_update()` emits — history shrinks through `add_messages`, never by mutation |
| `resolve_phase()` | Deterministic `AgentState → "planning" \| "executing" \| "finishing" \| None`; explicit metadata hint wins |
| `CompositeToolProvider(phase_capability_map=...)` | Intersects requested capabilities with the phase allowlist — narrows, never widens |
| `FakeToolProvider` | Deterministic `ToolProviderPort` double: static `agent → tools` mapping, ignores capabilities/phase |
| `ContextCompactor.compact_list()` | Position-based compaction for ephemeral message lists without stable ids |
| `Settings.context_compaction_*` | `enabled` (default `False`), `strategy` (default `truncate`), `max_messages=60`, `keep_recent=10`, `token_threshold`, `min_interval_messages=20` |

## Run it

```bash
uv run jupyter lab notebooks/loop_hardening.ipynb
uv run python examples/loop_hardening.py   # from the prismal repo
```

No API key is required — the default `truncate` strategy never calls an LLM and the gating demo uses fake providers. The notebook's final run cell ships commented out as `# await main()`; the `nest_asyncio.apply()` in the first cell lets the coroutine run directly on Jupyter's event loop.

## Related

- [Runtime hardening](runtime_hardening.md) — the Phase H runaway guard that stops loops compaction alone cannot save
- [Budget governance](budget_governance.md) — token ceilings and the `BudgetGuard` that can also trigger compaction
- [Custom tool providers](tool_provider_custom.md) — the `ToolProviderPort` seam that phase gating composes over
- [Memory management](memory_management.md) — the longer-horizon complement to in-loop context compaction
