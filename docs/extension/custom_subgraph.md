# PrismalStateGraphBuilder

> **Notebook:** [`notebooks/extension/custom_subgraph.ipynb`](../../notebooks/extension/custom_subgraph.ipynb) · **Example:** [`examples/extension/custom_subgraph.py`](https://github.com/prismal-ai/prismal/blob/main/examples/extension/custom_subgraph.py)

`PrismalStateGraphBuilder` is the fluent API of the Fase X extension surface for assembling whole subgraphs — nodes, edges, conditional routing, entry point — on top of `StateGraph[AgentState]` without forking the framework. Where [`@prismal_node`](custom_node.md) hardens a single function, the builder composes several of them into a pipeline, and it applies the same middleware automatically: any plain callable you pass to `add_node()` that lacks the `__prismal_node__` attribute is auto-wrapped with `@prismal_node` using the builder defaults, so every node in your graph gets sanitization, tracing, logging, audit, and error mapping whether you decorated it yourself or not.

The example builds a two-node pipeline (`classify → respond`), compiles it, and runs it end-to-end against a hand-built `AgentState` — entirely offline, with no LLM calls. This is the pattern third-party plugins use to ship pip-installable subgraphs that the supervisor can route to.

## What it demonstrates

- Declaring a mini-pipeline with the fluent builder: `add_node`, `add_edge`, `set_entry_point`, using `"__end__"` as the terminal target.
- Auto-wrapping: `classify` and `respond` are plain coroutines with no decorator — `add_node()` wraps them with `@prismal_node` transparently.
- The two compile targets: `compile()` returns a registrable `SubgraphDefinition`; `compile_raw()` returns a runnable `CompiledStateGraph`.
- Passing per-node options (`capabilities=["general"]`) through `add_node()` into the generated decorator.
- Executing the compiled graph with `ainvoke()` on an initial state and reading the final messages.

## How it works

### Define plain node functions

Both nodes are ordinary coroutines following the `async (state) -> state_update` contract — no decorator in sight:

```python
async def classify(state: dict) -> dict:
    text = state["messages"][-1].content
    return {"metadata": {"classify": {"len": len(text)}}}


async def respond(state: dict) -> dict:
    return {"messages": [AIMessage(content="Handled by the custom subgraph.")]}
```

`classify` writes only into `state["metadata"]`; `respond` appends an `AIMessage` (the `messages` field uses the `add_messages` reducer, so returned messages are appended, not replaced).

### Assemble the topology

```python
builder = PrismalStateGraphBuilder("demo_pipeline")
builder.add_node("classify", classify, capabilities=["general"])
builder.add_node("respond", respond)
builder.add_edge("classify", "respond")
builder.add_edge("respond", "__end__")
builder.set_entry_point("classify")
```

Because neither function carries `__prismal_node__`, `add_node()` wraps each with `@prismal_node` on the spot, applying `BuilderDefaults` (security `"standard"`, audit on) plus any per-node overrides — here `capabilities=["general"]` for `classify`, which also registers that capability in `tool_registry.DEFAULT_CAPABILITY_MAP`. If you pass an already-decorated node instead, it is used unchanged and the kwargs are ignored. The result: every node runs inside the middleware chain (outermost to innermost: error_mapping → OTel span → logger bind → security → audit → retry → timeout → user fn), while your functions stay plain coroutines.

The builder also validates before compiling: a missing entry point, an edge to an unknown node, or a duplicate node name raises `ValueError` with a precise message — errors surface at build time, not mid-run.

### Compile: definition vs. runnable graph

```python
# compile() -> SubgraphDefinition (registrable); compile_raw() -> runnable graph.
compiled = builder.compile_raw()
```

The builder has two exits, and choosing the right one matters:

- **`compile()`** returns a real `SubgraphDefinition` — the framework's declarative record (`name`, `nodes`, `edges`, `conditional_edges`, `entry_point`). This is what you hand to `SubgraphRegistry` (for example via `register_sync()` from a plugin entry point) so the supervisor can discover and route to your pipeline. It is a description, not an executable.
- **`compile_raw()`** is the escape hatch: it builds and compiles an actual `StateGraph[AgentState]` and returns a `CompiledStateGraph` you can `ainvoke` immediately.

The example uses `compile_raw()` because it wants to run the pipeline right here; the [plugin discovery demo](discover_plugins_demo.md) uses `compile()` because it wants to register it.

### Run it end-to-end

```python
state = create_initial_state(session_id="demo")
state["messages"] = [HumanMessage(content="Please summarise this.")]
result = await compiled.ainvoke(state)
print("final message:", result["messages"][-1].content)
```

LangGraph executes `classify`, follows the edge to `respond`, and reaches `END`. The final state contains both the original `HumanMessage` and the appended `AIMessage` ("Handled by the custom subgraph."), plus the `classify` metadata — proof that both nodes ran through the full middleware chain. In the notebook, the first cell applies `nest_asyncio`, so you can `await main()` directly in a cell.

### Beyond linear pipelines

The demo stays linear on purpose, but the builder covers the topologies real subgraphs need:

```python
builder.add_conditional_edges(
    "classify",
    lambda state: state["metadata"]["classify"]["route"],
    {"long": "summarise", "short": "respond"},
)
builder.add_supervisor_node(route_fn, valid_next=["summarise", "respond"])
builder.add_security_layer(at="entry")
```

- `add_conditional_edges()` routes from a node through a decision function plus a key-to-node mapping.
- `add_supervisor_node()` adds a routing node that writes `next_agent` and validates the decision against `valid_next` — deliberately *not* middleware-wrapped, so an out-of-range route fails loudly as a `ValueError` instead of being softened into an error state.
- `add_security_layer(at="entry" | "exit")` inserts a dedicated `InputSanitizer` node at the subgraph border, rewiring the entry point or every `__end__` edge accordingly — useful when a subgraph receives content from an external trust boundary.

## Key API

| Symbol | Role |
|---|---|
| `PrismalStateGraphBuilder` | Fluent builder over `StateGraph[AgentState]` with prismal middleware defaults. |
| `add_node(name, fn, **opts)` | Registers a node; auto-wraps plain callables lacking `__prismal_node__` with `@prismal_node`. |
| `add_edge(from_, to)` | Linear edge; `"__end__"` routes to LangGraph's `END`. |
| `add_conditional_edges(from_, decision_fn, mapping)` | Conditional routing from a node via a decision function. |
| `set_entry_point(name)` | Declares where execution starts. |
| `compile()` | Validates and returns a `SubgraphDefinition` (nodes/edges/conditional_edges/entry_point) for registration. |
| `compile_raw()` | Escape hatch: validates and returns a runnable `CompiledStateGraph`. |
| `BuilderDefaults` | Security/audit/timeout/retry defaults applied to auto-wrapped nodes. |
| `SubgraphDefinition` | Declarative subgraph record consumed by `SubgraphRegistry`. |

## Run it

```bash
uv run jupyter lab notebooks/extension/custom_subgraph.ipynb
uv run python examples/extension/custom_subgraph.py   # from the prismal repo
```

Runs fully offline on a hard-coded `AgentState` — no API keys required.

## Related

- [The `@prismal_node` decorator](custom_node.md) — what auto-wrapping puts around each node.
- [discover_plugins()](discover_plugins_demo.md) — register a builder-compiled `SubgraphDefinition` from a plugin entry point.
- [Visualizing graphs](../visualize_graphs.md) — render builders and `SubgraphDefinition`s as Mermaid diagrams.
- [Supervisor quickstart](../supervisor_quickstart.md) — how subgraphs fit into the main supervisor graph.
