# Exposing prismal as an A2A Agent — the Inbound Server Handler (Phase I)

> **Notebook:** [`notebooks/a2a_server.ipynb`](../notebooks/a2a_server.ipynb) · **Example:** [`examples/a2a_server.py`](https://github.com/prismal-ai/prismal/blob/main/examples/a2a_server.py)

Phase I adds bidirectional **A2A (Agent2Agent)** interoperability to prismal. Where MCP lets the framework consume *tools*, A2A operates one level up: it lets whole *agents* talk to each other over a standard wire protocol — `prismal/mcp/` stays for tools, `prismal/a2a/` is the agent-level complement. The layer is opt-in via `settings.a2a_enabled` (default `False`); with the flag off the compiled supervisor graph is byte-for-byte unchanged, and the HTTP/SSE dependencies live behind the `[a2a]` extra with deferred imports.

This article covers the **inbound** half: publishing a machine-readable Agent Card at `/.well-known/agent-card.json` and answering A2A JSON-RPC calls through `A2AServerHandler`, which maps each remote task onto the prismal graph. Because every inbound message crosses a trust boundary, the handler L1-sanitizes it (`InputSanitizer`) before it touches `AgentState`, and audits every task hash-first (`a2a.inbound`) — never the content.

## What it demonstrates

- Building the Agent Card from a capability registry with `build_agent_card` — one `AgentSkill` per published capability, camelCase wire format.
- Dispatching a `message/send` JSON-RPC request through `A2AServerHandler.handle_rpc` onto a graph, and reading back the completed `A2ATask` with its artifacts.
- Streaming the same task as Server-Sent Events with `stream_rpc` (SSE `data:` lines).
- The strict-mode auth gate: an unauthenticated call is rejected with JSON-RPC error `-32001` before any graph work happens.
- The substitution that makes it run offline: a tiny `EchoGraph` stands in for the real compiled supervisor graph, so no LLM, web server, or network is needed.

## How it works

### The trust boundary and the graph seam

`A2AServerHandler` is framework-agnostic: it does not own an HTTP server. In production the host (`prismal-server`) mounts the routes (`GET /.well-known/agent-card.json`, `POST /a2a`), validates the caller's credentials into an `AuthContext`, and passes `await get_async_compiled_graph()` as the graph. The handler's internal `_run_task` pipeline is: **sanitize the incoming text → invoke the graph with `thread_id=task_id` → yield the answer as an `A2AArtifact` → audit `a2a.inbound`**. Using the task id as the checkpoint thread id means a multi-turn A2A task naturally maps onto one conversation thread.

The notebook keeps all of that observable by replacing the graph with an in-process echo:

```python
class EchoGraph:
    """Stand-in for the compiled supervisor graph."""

    async def ainvoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        text = state["messages"][-1].content
        return {"messages": [*state["messages"], AIMessage(content=f"prismal handled: {text}")]}
```

Anything with an `ainvoke(state, config)` coroutine satisfies the seam, which is exactly why the demo needs no API key.

### The Agent Card

Discovery starts at the well-known URL. `build_agent_card(settings, registry)` derives one `AgentSkill` per entry in a capability registry (a mapping like `tool_registry.DEFAULT_CAPABILITY_MAP`); `settings.a2a_published_skills` acts as an allowlist (empty = publish everything). The card tenant-scopes its advertised URL per `org_id`, embeds a DID when the identity settings provide one (`did:web`), advertises media output modes when `multimodal_enabled`, and is cached per (settings fingerprint, org) — `clear_agent_card_cache()` drops it on reload.

```python
settings = Settings(
    a2a_enabled=True,
    a2a_inbound_enabled=True,
    a2a_base_url="https://prismal.example.com/a2a",
    a2a_strict=True,
)

card = build_agent_card(settings, REGISTRY)
print(json.dumps(card.model_dump(by_alias=True, exclude_none=True), indent=2))
```

Note `model_dump(by_alias=True)`: the A2A v0.3.x models keep snake_case Python attributes but serialize camelCase wire keys (`protocolVersion`, `inputModes`, `messageId`, …) via Pydantic's `alias_generator=to_camel`, and accept either form on input thanks to `populate_by_name`.

### JSON-RPC request/response

An A2A call is plain JSON-RPC 2.0. The notebook builds the envelope by hand so the wire shape stays visible:

```python
def _request(text: str) -> dict[str, Any]:
    msg = A2AMessage(role="user", parts=[A2APart(kind="text", text=text)], message_id="m1")
    return {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "message/send",
        "params": {"message": msg.model_dump(by_alias=True), "skillId": "research"},
    }
```

`handle_rpc` dispatches three methods — `message/send` (run the graph, return the completed `A2ATask`), `tasks/get` (fetch a stored task), and `tasks/cancel` (mark it canceled). The response for `message/send` carries the task's `history` (the sanitized incoming message) and `artifacts` (the agent's answer):

```python
handler = A2AServerHandler(EchoGraph(), settings=settings)
auth = AuthContext(authenticated=True, subject="acme-orchestrator")

response = await handler.handle_rpc(_request("summarize Q3 revenue"), auth_ctx=auth)
print("message/send →", json.dumps(response["result"], indent=2))
```

### SSE streaming

`stream_rpc` is the streaming twin: an async generator of SSE `data:` lines the host forwards to the caller. Each artifact arrives as a `{"kind": "artifact", ...}` result line, followed by a final `{"kind": "status", "status": "completed"}`:

```python
async for line in handler.stream_rpc(_request("stream this"), auth_ctx=auth):
    print("  ", line.strip())
```

### The strict-mode auth gate

The handler never validates credentials itself — the host does that and passes an `AuthContext(authenticated=..., subject=..., did=...)`. With `a2a_strict=True`, a missing or unauthenticated context is rejected with JSON-RPC error `-32001` ("authentication required") and the refusal is audited, before any sanitization or graph invocation:

```python
denied = await handler.handle_rpc(_request("no auth"), auth_ctx=None)
print("unauthenticated (strict) →", denied["error"])
```

With `a2a_strict=False` (the default), unauthenticated calls are allowed through — useful for local development, but strict mode is the intended posture for anything reachable from another organization.

### Task lifecycle and audit trail

`message/send` is only the first of the three dispatched methods. The handler stores each completed `A2ATask` keyed by its id, so a caller can later poll it with `tasks/get` or mark it canceled with `tasks/cancel` (an unknown id yields JSON-RPC error `-32004`, "task not found"). Every state transition — completion, cancellation, and even the strict-mode refusal above — lands in the append-only audit log as an `a2a.inbound` event carrying only identifiers and status (`task_id`, `skill`, `status`), never the message content. That hash-first discipline is the same one the rest of the security layer follows: the audit trail proves *that* an exchange happened and how it ended, without becoming a second copy of the data that crossed the boundary.

## Key API

| Symbol | Role |
|---|---|
| `A2AServerHandler` | Inbound handler: maps A2A JSON-RPC (`message/send`, `tasks/get`, `tasks/cancel`) onto the prismal graph; `handle_rpc` (request/response) + `stream_rpc` (SSE). |
| `build_agent_card` | Derives the Agent Card served at `/.well-known/agent-card.json` from a capability registry; allowlist, tenant URL, DID, per-(settings, org) cache. |
| `AuthContext` | Frozen caller identity the host passes after validating auth; gate for `a2a_strict` mode (`-32001` when unauthenticated). |
| `A2AMessage` / `A2APart` | A2A v0.3.x wire models for a task turn and its content parts (camelCase aliases, `model_dump(by_alias=True)`). |
| `A2ATask` / `A2AArtifact` | The stateful unit of work returned by `message/send` and the produced outputs it carries. |
| `Settings` | The `a2a_*` block: `a2a_enabled`, `a2a_inbound_enabled`, `a2a_base_url`, `a2a_published_skills`, `a2a_strict`. |
| `EchoGraph` (demo) | Offline stand-in for `await get_async_compiled_graph()` — anything with `ainvoke(state, config)` fits the seam. |

## Run it

```bash
uv run jupyter lab notebooks/a2a_server.ipynb
uv run python examples/a2a_server.py   # from the prismal repo
```

No API key is required — the demo runs fully offline with the injected `EchoGraph`. In the notebook, the final run cell is the commented `# await main()`: uncomment it and execute (it is `async`; `nest_asyncio.apply()` in the first cell makes `await` work inside Jupyter's event loop).

## Related

- [Delegating to a remote A2A agent as a graph node](a2a_remote_node.md) — the outbound half of Phase I.
- [Remote A2A agents as tools](a2a_tool_provider.md) — surfacing remote skills through the Fase Y tool port.
- [Agent identity (DIDs)](agent_identity.md) — where the `did:web` embedded in the Agent Card comes from.
- [Supervisor quickstart](supervisor_quickstart.md) — the real graph the handler drives in production.
