# Migrating LangChain chains with LangChainRunnableAdapter

> **Notebook:** [`notebooks/extension/langchain_migration.ipynb`](../../notebooks/extension/langchain_migration.ipynb) · **Example:** [`examples/extension/langchain_migration.py`](https://github.com/prismal-ai/prismal/blob/main/examples/extension/langchain_migration.py)

Most teams adopting prismal already have working LangChain code — LCEL chains, `Runnable` pipelines, `AgentExecutor` agents. The Fase X extension surface offers a migration bridge instead of a rewrite: `LangChainRunnableAdapter` converts any LangChain `Runnable` or `AgentExecutor` into a prismal graph node with `as_node(name=..., capabilities=...)`, handling the input/output mapping between prismal's `state["messages"]` and the runnable's own signature automatically. The result is a `@prismal_node`-wrapped callable, so your legacy chain inherits the full middleware chain — sanitization, tracing, logging, audit, timeout, error mapping — and slots into `PrismalStateGraphBuilder` or the supervisor graph like any native node.

The example keeps the "legacy" side minimal: a `RunnableLambda` standing in for your existing chain. Swap in a real LCEL pipeline or `AgentExecutor` and nothing else changes.

## What it demonstrates

- Wrapping an existing LangChain `Runnable` (here a `RunnableLambda`) without modifying it — the legacy code keeps working unchanged.
- Choosing an input mapping (`input_mapping="messages"`) so the runnable receives `state["messages"]` as a message list; `"auto"` and `"input_dict"` cover the `AgentExecutor` shape.
- Converting the adapter into a graph node with `as_node(name="legacy_research", capabilities=["research"])` — which also registers those capabilities in `tool_registry.DEFAULT_CAPABILITY_MAP`.
- The security upgrade migration buys for free: the adapted node runs inside the `@prismal_node` middleware chain (sanitization, tracing, audit, error mapping).
- Invoking the node directly with a hand-built `AgentState` and reading the mapped-back `state_update`.
- Automatic output normalization: whatever the runnable returns (message, string, dict) comes back as `{"messages": [AIMessage(...)]}`.

## How it works

### Wrap the legacy chain

```python
from langchain_core.runnables import RunnableLambda

from prismal.agents.extension import LangChainRunnableAdapter

# Pretend this is your existing LangChain chain / AgentExecutor.
legacy_chain = RunnableLambda(
    lambda msgs: AIMessage(content=f"legacy says: {msgs[-1].content}")
)

adapter = LangChainRunnableAdapter(legacy_chain, input_mapping="messages")
```

The adapter accepts anything with an `ainvoke` method — a plain `Runnable`, an LCEL composition, or an `AgentExecutor` — and rejects everything else early with a `LangChainAdapterError` (raised both for objects without `ainvoke` and for an unrecognised `input_mapping`). It never touches the wrapped object; migration is purely additive, and the original chain keeps working wherever it is used today.

The example passes `input_mapping="messages"` explicitly because a `RunnableLambda` gives the auto-detector nothing to inspect. With a real `AgentExecutor` you would normally leave the default `"auto"`, which detects the `input_keys` attribute and switches to the dict-shaped input automatically.

### Input and output mapping

The core problem the adapter solves is that prismal nodes speak `AgentState` while LangChain runnables each have their own input shape. Three `input_mapping` modes cover the common cases:

- `"messages"` — pass `state["messages"]` as a list of `BaseMessage` (used here, right for chat-style chains).
- `"input_dict"` — pass `{"input": last_user_message, "chat_history": prior_messages}` (the `AgentExecutor` convention).
- `"auto"` (the default) — detect: runnables exposing `input_keys` (AgentExecutor-style) get `"input_dict"`, everything else gets `"messages"`.

On the way back out, the adapter normalizes whatever the runnable produced into a valid `state_update`: an `AIMessage` passes through, a `str` is wrapped in one, and a `dict` has its text extracted (via an explicit `output_key`, else the conventional `"output"` key). Either way the node returns `{"messages": [AIMessage(...)]}`, which the `add_messages` reducer appends to graph state. A failing runnable surfaces as a `LangChainAdapterError` with the original exception chained.

For an `AgentExecutor`, the whole migration is typically:

```python
adapter = LangChainRunnableAdapter(my_agent_executor)   # "auto" detects input_keys
node = adapter.as_node(name="legacy_agent", capabilities=["research"])
```

The executor receives `{"input": ..., "chat_history": [...]}` and its `{"output": ...}` result is lifted into an `AIMessage` — no glue code on your side.

### Promote it to a prismal node

```python
node = adapter.as_node(name="legacy_research", capabilities=["research"])

state = create_initial_state(session_id="demo")
state["messages"] = [HumanMessage(content="hello from prismal")]
update = await node(state)
print("node output:", update["messages"][-1].content)
```

`as_node()` is where the adapter meets the rest of the extension surface: it wraps the mapped invocation with `@prismal_node`, so the legacy chain now runs inside the standard middleware chain (outermost to innermost: error_mapping → OTel span → logger bind → security → audit → retry → timeout → the adapted call). The `"standard"` security level sanitizes the last user message with `InputSanitizer` before the chain sees it, every invocation is audit-logged, and failures map to `NodeExecutionError` — guarantees your original LangChain code never had, gained without editing it. Passing `capabilities=["research"]` registers the node in `tool_registry.DEFAULT_CAPABILITY_MAP`, so the tool-provider layer can serve it research tools when it runs inside a graph.

The returned `node` is an ordinary `NodeFn` carrying `__prismal_node__`, which means `PrismalStateGraphBuilder.add_node("legacy_research", node)` uses it as-is (no re-wrapping), and it can equally be exposed as a `prismal.nodes` plugin entry point. The example invokes it directly, and the mock chain echoes the input: `legacy says: hello from prismal`. In the notebook, the first cell applies `nest_asyncio`, so you can `await main()` directly in a cell.

Two escape hatches are worth knowing when debugging a migration. `adapter.ainvoke(state)` runs the map-in → runnable → map-out cycle *without* the middleware, so you can isolate mapping problems from security or timeout behaviour. And `as_node()` accepts `security="off"` and `timeout_s=...` overrides when a legacy chain genuinely cannot tolerate the standard wrapping — though `"standard"` should be the default posture for anything that touches user input.

### A pragmatic migration path

This adapter turns migration into an incremental process rather than a rewrite:

1. **Wrap** each existing chain or executor in a `LangChainRunnableAdapter` and confirm the mapping with a direct `adapter.ainvoke(state)` call.
2. **Promote** it with `as_node(...)`, choosing a name, capabilities, and security level — the chain now runs with prismal's guarantees.
3. **Compose** the nodes into a pipeline with `PrismalStateGraphBuilder`, compile to a `SubgraphDefinition`, and register it so the supervisor can route to it.
4. **Ship** it, optionally, as a `prismal.nodes` or `prismal.subgraphs` plugin entry point so other installations pick it up via `discover_plugins()`.
5. **Rewrite** internals as native nodes later — or never; adapted nodes are first-class citizens.

Because everything is imported from `prismal.agents.extension` (with raw LangGraph symbols available via `prismal.langgraph`), the integration is version-stable across prismal releases: the extension surface is the supported contract, so upgrading prismal does not require chasing `langgraph.*` import churn in your migrated code.

## Key API

| Symbol | Role |
|---|---|
| `LangChainRunnableAdapter(runnable, *, input_mapping, output_key)` | Wraps any LangChain `Runnable`/`AgentExecutor`; validates it exposes `ainvoke`. |
| `InputMapping` | `"auto" \| "messages" \| "input_dict"` — how `AgentState` is mapped to the runnable's input. |
| `as_node(name=, capabilities=, security=, timeout_s=)` | Returns a `@prismal_node`-wrapped callable ready for `add_node()`. |
| `adapter.ainvoke(state)` | The raw map-in → run → map-out step, without middleware (handy in tests). |
| `LangChainAdapterError` | Raised for non-runnables, bad mappings, or a failing wrapped invocation. |
| `prismal_node` | The decorator `as_node()` applies — middleware chain plus capability registration. |
| `create_initial_state()` | Builds the `AgentState` used to exercise the node directly. |
| `RunnableLambda` | Stand-in for the legacy chain in this demo. |
| `PrismalStateGraphBuilder.add_node()` | Accepts the adapted node as-is (it already carries `__prismal_node__`, so no re-wrapping). |
| `tool_registry.DEFAULT_CAPABILITY_MAP` | Where `as_node(capabilities=...)` registers the node's capabilities as a side effect. |

The adapter itself is intentionally thin: input mapping, output normalization, and error translation. Everything else — security, observability, resilience — comes from the shared `@prismal_node` middleware, so adapted and native nodes are indistinguishable to the rest of the framework.

## Run it

```bash
uv run jupyter lab notebooks/extension/langchain_migration.ipynb
uv run python examples/extension/langchain_migration.py   # from the prismal repo
```

Runs fully offline with an inline mock Runnable — no API keys required. In the notebook, run the demo with `await main()`; the first cell applies `nest_asyncio` so async calls work directly in cells.

To try it against a real chain, replace the `RunnableLambda` with any LCEL pipeline or `AgentExecutor` you already have — the adapter call and `as_node()` line stay identical.

## Related

- [The `@prismal_node` decorator](custom_node.md) — the middleware chain `as_node()` puts around your chain.
- [PrismalStateGraphBuilder](custom_subgraph.md) — composing adapted nodes into a subgraph.
- [Custom tool providers](../tool_provider_custom.md) — how the registered capabilities resolve to tools.
- [Node type safety](../node_typesafety.md) — adding I/O contracts to nodes, adapted or native.
