# Delegating to a Remote A2A Agent — Outbound Node and Tools (Phase I)

> **Notebook:** [`notebooks/a2a_remote_node.ipynb`](../notebooks/a2a_remote_node.ipynb) · **Example:** [`examples/a2a_remote_node.py`](https://github.com/prismal-ai/prismal/blob/main/examples/a2a_remote_node.py)

Phase I gives prismal bidirectional **A2A (Agent2Agent)** interoperability — the agent-level complement to MCP's tool-level interop (`prismal/mcp/` stays for tools, `prismal/a2a/` adds agents). This article covers the **outbound** half: consuming a remote A2A agent from inside a prismal graph. Two integration shapes are shown — the remote agent as a *graph node* (`A2AAgentNode.as_node()`, the A2A analogue of `LangChainRunnableAdapter`) and the remote agent's skills as *tools* (`A2AToolProvider`, conforming to the Fase Y `ToolProviderPort`).

The layer is opt-in (`settings.a2a_enabled`, default `False`; `[a2a]` extra with deferred HTTP/SSE imports) and treats every remote answer as untrusted: artifacts are L1-sanitized (`InputSanitizer`) before they touch `AgentState`, and every outbound task is audited (`a2a.outbound`) without content. A remote failure never aborts the graph — the node degrades to a graceful `metadata.a2a.error` update.

## What it demonstrates

- Wrapping a remote agent described by an `AgentCard` as a prismal graph node with `A2AAgentNode(...).as_node(name=...)` and running it over a minimal state.
- How the remote artifacts are sanitized and merged into `state["messages"]`, with delegation metadata recorded under `state["metadata"]["a2a"]`.
- Surfacing the same remote agent's skills as `a2a__{agent}__{skill}` tools through `A2AToolProvider.get_tools()` and invoking one.
- The port seams that make it run offline: a `FakeRemoteClient` stands in for `A2AClient` (no network) and a `FakeManager` stands in for `A2AConnectionManager` (no allowlist, no pool).

## How it works

### The remote agent, described by its card

Everything outbound starts from an `AgentCard` — normally discovered from the remote's `/.well-known/agent-card.json`, here built in code so no discovery round-trip is needed. The card names the endpoint and enumerates the skills the agent publishes:

```python
CARD = AgentCard(
    name="billing",
    description="Remote billing agent",
    url="https://billing.acme/a2a",
    version="1.0.0",
    skills=[AgentSkill(id="create_invoice", name="Create Invoice", description="Issue an invoice")],
)
```

These are A2A v0.3.x Pydantic models: snake_case attributes, camelCase wire aliases (`protocolVersion`, `messageId`, …) via `alias_generator=to_camel`, serialized with `model_dump(by_alias=True)`.

### Faking the transport at the client seam

`A2AAgentNode` and `A2AToolProvider` never speak HTTP themselves — they ask a client for `send_task(message, skill_id=...)`, an async iterator of `A2AArtifact`s. That is the substitution point. The notebook injects an in-process double that yields one artifact and touches no network:

```python
class FakeRemoteClient:
    """Stand-in for A2AClient — yields a single artifact, no network."""

    async def send_task(
        self, message: A2AMessage, *, skill_id: str | None = None
    ) -> AsyncIterator[A2AArtifact]:
        text = message.parts[0].text if message.parts else ""
        yield A2AArtifact(
            artifact_id="a1",
            parts=[A2APart(kind="text", text=f"invoice created for: {text}")],
        )
```

In production the real `A2AClient` discovers the card, sends the task as JSON-RPC `message/send`, consumes the SSE stream of artifacts, supports `cancel`, and handles `bearer` / `oauth2_client_credentials` auth with token caching. Clients are obtained through `A2AConnectionManager`, which enforces the fnmatch outbound allowlist (`a2a_outbound_allowlist`, deny-all in strict mode) and pools connections — the mirror of `mcp/connection.py`. The notebook's `FakeManager` implements just its `get_client(url)` surface:

```python
class FakeManager:
    def __init__(self, client: FakeRemoteClient) -> None:
        self._client = client

    async def get_client(self, url: str) -> FakeRemoteClient:
        return self._client
```

### Shape 1 — the remote agent as a graph node

`as_node()` returns an `async (state) -> state_update` wrapped with `@prismal_node`, so the remote delegate gets the standard middleware chain (OTel span, audit, security, timeout) for free and can be dropped into any `StateGraph` via `add_node`:

```python
node = A2AAgentNode(CARD, client=FakeRemoteClient(), skill_id="create_invoice").as_node(
    name="billing_agent"
)
state = {"messages": [HumanMessage(content="bill customer #42 for $99")]}
update = await node(state)
answer = next(m for m in update["messages"] if isinstance(m, AIMessage)).content
```

Internally the node takes the last user turn from `state["messages"]`, wraps it in an `A2AMessage`, streams the remote artifacts, **sanitizes** the concatenated text through `InputSanitizer`, and returns it as an `AIMessage` plus a metadata record:

- success → `{"metadata": {"a2a": {"error": False, "agent": <url>, "artifacts": N}}}`
- remote failure → an apologetic `AIMessage` plus `{"a2a": {"error": True, "agent": <url>}}` — the graph keeps running.

Both outcomes are audited as `a2a.outbound` events (agent URL, skill, status — never the content).

### Shape 2 — the remote skills as tools

The same card can instead feed `A2AToolProvider`, which conforms to the Fase Y `ToolProviderPort` (`get_tools` is sync and never raises). Each published skill becomes a `StructuredTool` named `a2a__{agent}__{skill}` that any prismal agent can call like a local tool:

```python
provider = A2AToolProvider([CARD], manager=FakeManager(FakeRemoteClient()))
tools = provider.get_tools(agent_name="researcher")
print("exposed tools  →", [t.name for t in tools])       # ['a2a__billing__create_invoice']
result = await tools[0].ainvoke({"query": "invoice for ACME"})
```

Tool invocations route through the manager (allowlist + pool in production), and their results pass the same `InputSanitizer` before returning. Because the provider is a standard tool port, it composes with MCP, skills, and stub providers inside `CompositeToolProvider` — the subject of the companion article below.

### Node or tool — which shape to pick

The two shapes answer different design questions:

- **Node** (`A2AAgentNode.as_node`) makes the delegation an explicit step in your graph topology: the supervisor (or a subgraph edge) *decides* to route a turn to the remote agent, and the whole turn becomes its job. Pick this when the remote agent owns a phase of the workflow — a billing step, a compliance check, an external research pass.
- **Tool** (`A2AToolProvider`) leaves the decision to the LLM inside an existing agent: the remote skill appears next to local tools and gets called mid-reasoning like any other. Pick this when the remote capability is one option among many rather than a routing destination.

Both shapes share the same trust-boundary treatment (sanitize, audit, degrade gracefully) and the same transport seams, so switching between them is a wiring change, not a rewrite.

## Key API

| Symbol | Role |
|---|---|
| `A2AAgentNode` | Wraps a remote A2A agent as a graph node; `.as_node(name=...)` returns a `@prismal_node`-decorated callable (the A2A analogue of `LangChainRunnableAdapter`). |
| `A2AToolProvider` | Fase Y `ToolProviderPort` adapter: surfaces remote skills as `a2a__{agent}__{skill}` tools; sync `get_tools` never raises. |
| `A2AClient` (faked here) | Outbound transport: card discovery, `send_task` via JSON-RPC `message/send` + SSE, `cancel`, bearer/OAuth2 auth with token caching. |
| `A2AConnectionManager` (faked here) | fnmatch allowlist + deny-all-in-strict + client pool — the outbound mirror of `mcp/connection.py`. |
| `AgentCard` / `AgentSkill` | The remote agent's self-description and its published capabilities. |
| `A2AMessage` / `A2APart` / `A2AArtifact` | Wire models for the delegated turn and the streamed results. |

## Run it

```bash
uv run jupyter lab notebooks/a2a_remote_node.ipynb
uv run python examples/a2a_remote_node.py   # from the prismal repo
```

No API key is required — the demo runs fully offline with the injected fakes. In the notebook, the final run cell is the commented `# await main()`: uncomment it and execute (it is `async`; `nest_asyncio.apply()` in the first cell lets `await` run inside Jupyter's event loop).

## Related

- [Exposing prismal as an A2A agent](a2a_server.md) — the inbound half of Phase I.
- [Remote A2A agents as tools](a2a_tool_provider.md) — the provider shape in depth, composed with local tools.
- [Host-side tool provider wiring](tool_provider_host.md) — the Fase Y port this provider plugs into.
- [Composition root](composition_root.md) — where `build_runtime(..., a2a_agents=[...])` wires all of this for a host.
