# Remote A2A Agents as Tools — Provider Composition (Phase I + Fase Y)

> **Notebook:** [`notebooks/a2a_tool_provider.ipynb`](../notebooks/a2a_tool_provider.ipynb) · **Example:** [`examples/a2a_tool_provider.py`](https://github.com/prismal-ai/prismal/blob/main/examples/a2a_tool_provider.py)

This article sits at the intersection of two inversions. **Phase I** adds A2A (Agent2Agent) interoperability — the agent-level complement to MCP's tool-level interop (`prismal/mcp/` stays for tools, `prismal/a2a/` adds agents; opt-in via `settings.a2a_enabled`, `[a2a]` extra, deferred HTTP/SSE imports). **Fase Y** inverted tool resolution as a hexagonal port: the agent core never imports `prismal.mcp` or `prismal.skills` — it asks an injected `ToolProviderPort` (`get_tools(*, agent_name, capabilities) -> list[BaseTool]`, sync, must not raise).

`A2AToolProvider` is where they meet: it conforms to the Fase Y port while speaking the Phase I protocol, surfacing each skill of a remote A2A agent as a `a2a__{agent}__{skill}` tool. Because it is just another provider, it composes with MCP, skills, and stub providers inside `CompositeToolProvider` — remote agents become tools without the core knowing the difference. As always at a trust boundary, remote results are L1-sanitized (`InputSanitizer`) before being returned to the caller.

## What it demonstrates

- Building an `AgentCard` in code (no `/.well-known` discovery round-trip) with two published `AgentSkill`s carrying capability `tags`.
- Surfacing those skills as `a2a__{agent}__{skill}` tools via `A2AToolProvider.get_tools()`, including capability filtering against the skills' tags.
- Invoking one remote tool end-to-end through a fake in-process transport — the result comes back sanitized.
- Composing the A2A provider with a local `@tool` through `CompositeToolProvider` + `FakeToolProvider` (Fase Y), yielding one merged tool list for an agent.
- The production path: `build_runtime(settings, a2a_agents=[card])` performs the same wiring in the composition root, and `provider.prepare()` async-discovers URL-only agents.

## How it works

### The card: skills with capability tags

The provider consumes resolved `AgentCard`s (or plain URLs, resolved later by `prepare()`). Each `AgentSkill` carries `tags`, which are what the port's `capabilities` filter matches against:

```python
CARD = AgentCard(
    name="Research Hub",
    description="A remote research agent exposed over A2A.",
    url="https://research-hub.example/a2a",
    version="1.0.0",
    skills=[
        AgentSkill(
            id="web_search",
            name="Web search",
            description="Search the public web and return cited findings.",
            tags=["research"],
        ),
        AgentSkill(
            id="summarize",
            name="Summarize",
            description="Condense a document into key bullet points.",
            tags=["writing"],
        ),
    ],
)
```

These are A2A v0.3.x Pydantic models with camelCase wire aliases (`protocolVersion`, `inputModes`, …) via `alias_generator=to_camel`; on the wire they serialize with `model_dump(by_alias=True)`.

### Faking the transport

When a surfaced tool is invoked, the provider asks its `A2AConnectionManager` for a client and calls `send_task(message, skill_id=...)` — JSON-RPC `message/send` plus an SSE artifact stream in production. Both are seams, and the notebook substitutes in-process doubles so nothing leaves the interpreter:

```python
class _FakeA2AClient:
    """In-process stand-in for ``A2AClient`` — yields one text artifact per task."""

    async def send_task(
        self, message: A2AMessage, *, skill_id: str | None = None
    ) -> AsyncIterator[A2AArtifact]:
        query = (message.parts[0].text or "") if message.parts else ""
        yield A2AArtifact(
            artifact_id="art-1",
            parts=[A2APart(kind="text", text=f"[research-hub/{skill_id}] findings for: {query}")],
        )


class _FakeConnectionManager:
    """Stand-in for ``A2AConnectionManager`` — hands out the fake client."""

    async def get_client(self, url: str) -> _FakeA2AClient:
        return _FakeA2AClient("research-hub")
```

The real `A2AConnectionManager` additionally enforces the fnmatch outbound allowlist (`a2a_outbound_allowlist`, deny-all in strict mode) and pools clients — the mirror of `mcp/connection.py`.

### The port contract in action

`get_tools` honours the Fase Y contract: it is **sync**, it **never raises** (failures degrade to warnings and an empty list), and it filters by capabilities — here by intersecting the requested capability list with each skill's `tags`:

```python
provider = A2AToolProvider([CARD], manager=_FakeConnectionManager())

all_tools = provider.get_tools(agent_name="researcher")
# ['a2a__research_hub__web_search', 'a2a__research_hub__summarize']

research_only = provider.get_tools(agent_name="researcher", capabilities=["research"])
# ['a2a__research_hub__web_search']
```

Tool names follow `a2a__{agent}__{skill}` with the agent name slugified (`Research Hub` → `research_hub`), so they are stable, collision-resistant identifiers inside the global tool registry. Discovery, being network I/O, stays async: agents passed as URLs are only exposed after `await provider.prepare()` — typically run inside the async `build_runtime` — while resolved cards (as here) are exposed immediately.

### Invoking a remote skill as a tool

Calling the tool sends one A2A task through the (fake) transport, collects the text parts of the streamed artifacts, and returns them **sanitized** — remote content crosses a trust boundary, so it passes `InputSanitizer` before any agent sees it. A remote failure never propagates: the tool returns a descriptive `[a2a] remote agent ... unavailable` string instead of raising.

```python
answer = await research_only[0].ainvoke({"query": "quantum sensing startups"})
# [research-hub/web_search] findings for: quantum sensing startups
```

### Composition with local tools (Fase Y)

Because `A2AToolProvider` is just a `ToolProviderPort`, it slots into `CompositeToolProvider` alongside any other provider. The demo merges it with a `FakeToolProvider` carrying one local `@tool`:

```python
composite = CompositeToolProvider([provider, FakeToolProvider({"researcher": [local_notes]})])
merged = composite.get_tools(agent_name="researcher", capabilities=["research"])
# ['a2a__research_hub__web_search', 'local_notes']
```

In a real host you do not build this by hand — the composition root does the wiring when `a2a_enabled=True`:

```python
ctx = await build_runtime(settings, a2a_agents=[card])
```

`build_runtime` composes the same `A2AToolProvider` into the runtime tool port (alongside MCP and skills, under the global 120-tool cap) and awaits `prepare()` to discover any URL-only agents.

## Key API

| Symbol | Role |
|---|---|
| `A2AToolProvider` | Phase I adapter for the Fase Y `ToolProviderPort`: remote skills as `a2a__{agent}__{skill}` tools; sync `get_tools` never raises; async `prepare()` discovers URL-only agents. |
| `CompositeToolProvider` | Fase Y merger: combines multiple providers into one tool list per agent (priority order, name dedupe, global cap). |
| `FakeToolProvider` | Fase Y test double: a fixed `{agent: [tools]}` mapping — how you inject local tools in tests and demos. |
| `AgentCard` / `AgentSkill` | The remote agent's self-description; skill `tags` are what `capabilities` filters match. |
| `A2AMessage` / `A2APart` / `A2AArtifact` | Wire models for the delegated task and its streamed results (camelCase aliases). |
| `A2AConnectionManager` (faked here) | Allowlist + client pool for outbound A2A — the mirror of `mcp/connection.py`. |
| `build_runtime` | Composition-root entry point that performs this wiring for a host (`a2a_agents=[...]`). |

## Run it

```bash
uv run jupyter lab notebooks/a2a_tool_provider.ipynb
uv run python examples/a2a_tool_provider.py   # from the prismal repo
```

No API key is required — the demo runs fully offline with the injected fakes. In the notebook, the final run cell is the commented `# await main()`: uncomment it and execute (it is `async`; `nest_asyncio.apply()` in the first cell lets `await` run inside Jupyter's event loop).

## Related

- [Delegating to a remote A2A agent as a graph node](a2a_remote_node.md) — the node-shaped alternative to tools.
- [Exposing prismal as an A2A agent](a2a_server.md) — the inbound half of Phase I.
- [Host-side tool provider wiring](tool_provider_host.md) — the Fase Y default composition this provider joins.
- [Composition root](composition_root.md) — `build_runtime` assembling every port, A2A included.
