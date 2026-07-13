# Supervisor graph quickstart

> **Notebook:** [`notebooks/supervisor_quickstart.ipynb`](../notebooks/supervisor_quickstart.ipynb) · **Example:** [`examples/supervisor_quickstart.py`](https://github.com/prismal-ai/prismal/blob/main/examples/supervisor_quickstart.py)

The core of prismal is a LangGraph SUPERVISOR state machine: a `StateGraph[AgentState]` whose
central `supervisor` node inspects the conversation each turn, picks exactly one specialist agent
(`coder`, `researcher`, `rag_agent`, `planner`, and the rest of the 26-agent roster), runs it, and
receives control back. The loop repeats — supervisor → specialist → supervisor → … — until the
supervisor routes to `END` and the turn is over. Every other architecture in the framework
(hierarchical domains, swarms, subgraph pipelines) is layered on top of this machine, so this is
the right first notebook to read.

This walkthrough is deliberately offline. It exercises the two routing stages that decide where a
turn goes — deterministic regex intent matching first, LLM supervision second — then builds the
*real* compiled supervisor graph with an in-memory checkpointer and an injected fake tool
provider, and inspects its topology and the initial `AgentState`. Because actually invoking the
graph requires an LLM provider key, the demo stops short of `ainvoke` and shows you exactly what
that call would look like.

## What it demonstrates

- The supervisor → specialist → supervisor → `END` control loop that underpins all of prismal.
- Stage 1 routing: `match_intent()`, a stdlib-only regex matcher that short-circuits
  high-confidence intents (cron management, Tree of Thoughts, debate, ETL, multimodal, kokoro,
  skynet) before any LLM is consulted — including its educational-question denylist.
- Fase Y tool injection: agents never import MCP or skills directly; tools arrive through the
  injected `ToolProviderPort`, here a deterministic `FakeToolProvider`.
- Building and inspecting the real compiled supervisor graph offline with `build_supervisor_graph`
  and an `InMemorySaver` checkpointer.
- The shape of `create_initial_state()` — the `AgentState` you would pass to `graph.ainvoke`.

## How it works

### Stage 1 — deterministic intent routing

Before the supervisor ever asks an LLM, it runs the last user message through
`prismal.agents.intent_router.match_intent`. The module is pure stdlib (`re` + `unicodedata`),
side-effect-free, and normalizes text (lowercase, NFKD, accent stripping) so English and Spanish
utterances match the same patterns. When it recognizes a high-confidence intent it returns the
target node name and the LLM routing call is skipped entirely; when it returns `None`, the
supervisor falls through to LLM-based supervision.

```python
def demo_intent_routing() -> None:
    """Stage 1 — deterministic regex routing, no LLM and no I/O involved."""
    print("== match_intent(): deterministic supervisor short-circuit ==")
    for utterance in UTTERANCES:
        target = match_intent(utterance)
        label = target if target is not None else "(none — LLM supervision)"
        print(f"  {utterance!r:<48} -> {label}")
```

The notebook feeds it a batch covering every deterministic intent family:

```python
UTTERANCES: list[str] = [
    "Schedule a reminder every day at 9am",  # cron_manager (verb + recurrence)
    "List my cron jobs",                     # cron_manager (verb + noun)
    "Use tree of thoughts to plan the rollout",  # tot_agent
    "Hold a debate about microservices",     # debate_agent
    "Reach a consensus on the API design",   # debate_consensus
    "Run an ETL over the sales exports",     # data_etl
    "Describe this image of the dashboard",  # multimodal_pipeline
    "Transcribe this voice note",            # multimodal_pipeline
    "Deliberate on the migration plan",      # kokoro
    "Run a swarm over these five repositories",  # skynet
    "Explain how cron works",                # None — educational denylist
    "What is the capital of France?",        # None — falls through to the LLM
]
```

Two cases deliberately return `None`. "Explain how cron works" mentions cron but trips the
educational denylist (`explain`, `what is`, `how does`, …), which overrides any positive match —
asking *about* cron is not a request to *manage* cron jobs. And a generic question matches nothing,
so the LLM supervisor decides. Note that the supervisor also filters the matched intent against
`effective_valid_routes(...)`: an intent whose feature flag is off (say, `kokoro_enabled=False`)
falls back to LLM routing rather than targeting a node that is not in the graph.

### Stage 2 — inject a tool provider, then build the graph

Under Fase Y, tool resolution is inverted: the agent core asks an injected `ToolProviderPort`
rather than importing `prismal.mcp` or `prismal.skills`. The notebook installs a
`FakeToolProvider` so no MCP or skills I/O runs, then verifies the wiring through the same
`get_tools_for_agent()` call the real agent nodes use:

```python
@tool
def take_note(text: str) -> str:
    """Store a short research note (offline stand-in for a real tool)."""
    return f"noted: {text}"

provider = FakeToolProvider(mapping={"researcher": [take_note]})
set_tool_provider(provider)
researcher_tools = [t.name for t in get_tools_for_agent("researcher")]
```

The `researcher` gets `take_note`; `critic` gets nothing, because fixed-tool agents only ever see
stubs. This is the same seam tests use — inject a fake provider instead of patching registry
internals.

### Compiling and inspecting the supervisor graph

With tools injected, `build_supervisor_graph` compiles the full state machine. The notebook passes
an `InMemorySaver` so nothing touches disk:

```python
with tempfile.TemporaryDirectory() as tmp:
    graph = build_supervisor_graph(
        checkpoint_path=Path(tmp) / "checkpoints.db",
        checkpointer=InMemorySaver(),
    )
    nodes = sorted(graph.get_graph().nodes)
    print(f"== Supervisor graph compiled ({len(nodes)} nodes) ==")
```

Listing `graph.get_graph().nodes` shows the supervisor plus every specialist node, each with a
return edge back to the supervisor and a conditional edge fan-out from it. In async production
code you should prefer `await get_async_compiled_graph()`, which wires an `AsyncSqliteSaver`
(critical rule 3 of the framework) — the sync build here is only for topology inspection.

Finally, `create_initial_state(session_id=...)` shows the `AgentState` contract: `current_agent`,
`next_agent`, `session_id`, `created_at`, and an empty `messages` list (the only field with a
custom `add_messages` reducer). With provider keys configured, the next step would be
`await graph.ainvoke(state, config)` with a `thread_id` in the config so the checkpointer can
persist the conversation thread.

## Key API

| Symbol | Role |
|---|---|
| `prismal.agents.intent_router.match_intent` | Deterministic regex intent matcher; returns a node name or `None` (fall through to LLM routing) |
| `prismal.agents.graph.build_supervisor_graph` | Builds and compiles the LangGraph SUPERVISOR state machine (accepts a custom checkpointer) |
| `prismal.agents.graph.get_async_compiled_graph` | Cached async variant with `AsyncSqliteSaver` — what production async code should use |
| `prismal.agents.state.create_initial_state` | Creates a fresh `AgentState` TypedDict for a session |
| `prismal.agents.extension.providers.FakeToolProvider` | Deterministic `ToolProviderPort` for offline tests and notebooks |
| `prismal.agents.tool_registry.set_tool_provider` / `get_tools_for_agent` | Install a provider globally / resolve an agent's tools through it |

## Run it

```bash
uv run jupyter lab notebooks/supervisor_quickstart.ipynb
uv run python examples/supervisor_quickstart.py   # from the prismal repo
```

No API key required — the demo runs fully offline with injected fakes.

## Related

- [Hierarchical supervisor](hierarchical_supervisor.md) — domain orchestrators and network routing on top of the same loop
- [Parallel research fan-out](parallel_research_fanout.md) — the supervisor's Send()-based map-reduce branch
- [Custom tool provider](tool_provider_custom.md) — writing your own `ToolProviderPort`
- [Visualize graphs](visualize_graphs.md) — rendering the supervisor topology as Mermaid diagrams
