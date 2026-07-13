# The `@prismal_node` decorator

> **Notebook:** [`notebooks/extension/custom_node.ipynb`](../../notebooks/extension/custom_node.ipynb) · **Example:** [`examples/extension/custom_node.py`](https://github.com/prismal-ai/prismal/blob/main/examples/extension/custom_node.py)

This is the smallest possible entry point into prismal's public extension surface (Fase X), which makes LangGraph a first-class build target for your own code — no fork of the framework required. You write a plain `async (state) -> state_update` function, decorate it with `@prismal_node`, and the framework wraps it in the full production middleware chain: input sanitization, OpenTelemetry tracing, structured logging, retry/backoff, timeouts, audit logging, and error mapping. The decorated function is still an ordinary callable, so it drops into `PrismalStateGraphBuilder`, a raw `StateGraph`, or a plugin entry point unchanged.

Use `@prismal_node` whenever you want custom logic to run inside a prismal graph with the same security and observability guarantees as the framework's own 26 agents. The example keeps the business logic deliberately trivial — a mocked sentiment classifier with no LLM call — so you can see exactly what the decorator adds around your code.

## What it demonstrates

- Declaring a custom node with `@prismal_node(name=..., capabilities=...)` — a one-decorator upgrade from plain coroutine to production-grade graph node.
- The node contract: `async (state) -> state_update`, where the update is a partial `AgentState` dict merged back into graph state.
- The decorator's side effect: capabilities are registered in `tool_registry.DEFAULT_CAPABILITY_MAP`, so the tool-provider layer knows which tools this node may request.
- Inspecting the attached `__prismal_node__` metadata (`NodeMetadata`: name, capabilities, security level, audit flag, retry policy, timeout).
- Invoking the wrapped node directly against a hand-built state — no compiled graph, no LLM, no API keys.

## How it works

### Declare the node

The whole example is one decorated coroutine. The function body is your code; everything else is supplied by the decorator.

```python
from prismal.agents.extension import prismal_node
from prismal.agents.state import create_initial_state


@prismal_node(name="sentiment_classifier", capabilities=["general"])
async def sentiment_classifier(state: dict) -> dict:
    """Classify the last user message (mocked — no LLM call)."""
    text = state["messages"][-1].content.lower()
    label = "positive" if any(w in text for w in ("great", "love", "good")) else "neutral"
    return {"metadata": {"sentiment_classifier": {"label": label}}}
```

Two things are worth noting about the contract. First, the node reads from `state` and returns a *partial* update — here it writes only under `state["metadata"]`, leaving `messages` and everything else untouched. Second, importing from `prismal.agents.extension` (or `prismal.langgraph` for the raw LangGraph symbols) is the supported public contract: it guarantees version compatibility across prismal releases, unlike importing `langgraph.*` directly.

### What the decorator wraps around your function

`@prismal_node` builds a middleware pipeline around the user function. From outermost to innermost the chain is:

```
error_mapping → otel span → logger bind → security → audit → retry → timeout → user fn
```

- **security** (default level `"standard"`) runs `InputSanitizer` on the last user message before your code sees it; `"strict"` additionally permission-checks any `tool_call` in the output via the `ActionInterceptor` seam, and prompts assembled from user text go through `SecurePromptBuilder`.
- **otel + logger** give every invocation a span and a bound structured-log context.
- **retry** (opt-in via a `RetryPolicy`) and **timeout** (`timeout_s`, enforced with `asyncio.wait_for`) sit closest to your function, so retries happen before errors are mapped and the audit duration covers all attempts.
- **audit** writes an `AuditLogger.log_node` record per invocation (on by default).
- **error_mapping** converts failures into `NodeExecutionError` — by default returned as `{"metadata": {"error": {...}}}` rather than raised, unless you pass `raise_on_error=True`.

Your code stays a plain `async (state) -> state_update`; none of this leaks into the function body.

### Registration side effects

Decoration is not just wrapping. It also records an immutable `NodeMetadata` in the extension registry (queryable via `list_registered_nodes()` / `get_node_metadata(name)`), and — because the example passes `capabilities=["general"]` — registers those capabilities in `tool_registry.DEFAULT_CAPABILITY_MAP`. That map is how the injected `ToolProviderPort` later decides which tools to hand this node. Decoration is idempotent: applying `@prismal_node` to an already-decorated function returns it unchanged.

### Invoke it directly

Because the wrapped node is still a callable, you can exercise it without compiling a graph:

```python
async def main() -> None:
    state = create_initial_state(session_id="demo")
    state["messages"] = [HumanMessage(content="I love this framework")]

    update = await sentiment_classifier(state)
    print("state_update:", update)
    print("metadata:", sentiment_classifier.__prismal_node__)
```

The printed `state_update` is the partial dict your function returned (after passing through the full middleware chain), and `__prismal_node__` is the frozen `NodeMetadata` dataclass the decorator attached. The classifier is a keyword heuristic, so the run is deterministic and offline.

In the notebook, the first cell applies `nest_asyncio`, so you can simply `await main()` in a cell instead of `asyncio.run(main())`.

### Tuning resilience per node

The example uses the decorator defaults, but every middleware layer is configurable per node. A node that calls a flaky external service might declare:

```python
from prismal.agents.extension import RetryPolicy, prismal_node


@prismal_node(
    name="flaky_fetcher",
    capabilities=["research"],
    security="strict",
    retry=RetryPolicy(max_attempts=3, backoff_s=(0.1, 0.5, 1.0)),
    timeout_s=10.0,
    raise_on_error=True,
)
async def flaky_fetcher(state: dict) -> dict:
    ...
```

`retry=None` (the default) disables retries entirely; `timeout_s=None` disables the timeout; `raise_on_error=True` propagates `NodeExecutionError` to the graph instead of the default soft-fail `{"metadata": {"error": {...}}}` update. Because retry and timeout sit innermost in the chain, a timeout counts as one failed attempt that the retry layer may re-run.

## Key API

| Symbol | Role |
|---|---|
| `prismal_node` | Decorator that wraps an `async (state) -> state_update` in the middleware chain and registers its metadata. |
| `NodeMetadata` | Frozen dataclass attached as `__prismal_node__`: name, capabilities, security, audit, retry, timeout, source module. |
| `RetryPolicy` | Opt-in retry configuration (`max_attempts`, `backoff_s`, `retry_on`) for the retry middleware. |
| `SecurityLevel` | `"off" \| "standard" \| "strict"` — how much security middleware wraps the node. |
| `NodeFn` | Type alias for the node signature: `Callable[[AgentState], Awaitable[dict]]`. |
| `list_registered_nodes()` / `get_node_metadata()` | Query every node registered via the decorator. |
| `create_initial_state()` | Build a fresh `AgentState` for direct invocation in tests and demos. |
| `tool_registry.DEFAULT_CAPABILITY_MAP` | Where the decorator registers the node's capabilities as a side effect. |

## Run it

```bash
uv run jupyter lab notebooks/extension/custom_node.ipynb
uv run python examples/extension/custom_node.py   # from the prismal repo
```

Runs fully offline on a hard-coded message with a mocked classifier — no API keys required.

## Related

- [PrismalStateGraphBuilder](custom_subgraph.md) — wire decorated nodes into a compiled subgraph.
- [LangChainRunnableAdapter](langchain_migration.md) — turn existing LangChain Runnables into `@prismal_node` nodes.
- [Node type safety](../node_typesafety.md) — attach `input_model`/`output_model` contracts to nodes.
- [Custom tool providers](../tool_provider_custom.md) — how capabilities map to injected tools.
