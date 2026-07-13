# Host-style tool provider composition and injection (Fase Y)

> **Notebook:** [`notebooks/tool_provider_host.ipynb`](../notebooks/tool_provider_host.ipynb) · **Example:** [`examples/tool_provider_host.py`](https://github.com/prismal-ai/prismal/blob/main/examples/tool_provider_host.py)

Fase Y inverts tool resolution as a hexagonal port. Before the inversion, the agent core reached out to `prismal.mcp` and `prismal.skills` directly to gather tools; after it, the core only asks an injected `ToolProviderPort` — a sync `get_tools(*, agent_name, capabilities) -> list[BaseTool]` that must not raise — and never imports those packages again (an AST architecture test, `test_no_mcp_skills_imports.py`, enforces the boundary). The *host* — a server, an SDK, a script — becomes the party that decides where tools come from: it composes providers and injects the result.

This notebook plays the host role. It shows the two injection variants a real host uses in its lifespan: **variante A**, a single global injection with `set_tool_provider()` that every one of the 26 agent nodes resolves through unchanged, and **variante B**, a per-session provider carried in the LangGraph invocation config for multi-tenant deployments. By default it composes Skills + stubs only, so it runs offline and deterministically; setting `EXAMPLE_USE_MCP=1` switches to the full `build_default_tool_provider()`, which also connects the MCP servers declared in `config/mcp_servers.yaml`.

## What it demonstrates

- Composing the standard adapters — `SkillToolProvider` + `StubToolProvider` under a `CompositeToolProvider` — exactly the way a host lifespan does, with the full MCP composite one env var away.
- **Variante A (global):** injecting the composite once with `set_tool_provider()`, then resolving tools for several agents through the unchanged `get_tools_for_agent()` facade.
- The composite's merge policy in action: MCP → Skills → stubs priority, name dedupe, a hard `max_total=120` cap, and fixed-tool agents (`cron_manager`, `critic`) receiving stubs only.
- **Variante B (per-session):** binding a different `FakeToolProvider` per session via `config["configurable"]["tool_provider"]` and resolving with `get_tools_for_agent_ctx()` — two concurrent sessions see disjoint toolsets with no global state touched.

## How it works

### Imports: the port adapters and the registry facade

Everything the host needs lives in two modules: the provider adapters in `prismal.agents.extension` and the injection/resolution functions in `prismal.agents.tool_registry`.

```python
from prismal.agents.extension import (
    CompositeToolProvider,
    FakeToolProvider,
    SkillToolProvider,
    StubToolProvider,
    build_default_tool_provider,
)
from prismal.agents.tool_registry import (
    get_tools_for_agent,
    get_tools_for_agent_ctx,
    set_tool_provider,
)
```

The adapters are the *only* code allowed to import `prismal.mcp` / `prismal.skills` (they are the exemption in the architecture test): `McpToolProvider` caps its output at 60 tools, `SkillToolProvider` creates a fresh `SkillsManager` per call, `StubToolProvider` encapsulates the historical static `stub_map`, and `CompositeToolProvider` merges any number of sub-providers under the legacy policy. `FakeToolProvider` is the deterministic test double.

A tiny factory produces demo tools so the notebook stays self-contained:

```python
def _make_tool(name: str) -> StructuredTool:
    def _fn(query: str = "") -> str:
        return f"[{name}] {query}"

    return StructuredTool.from_function(func=_fn, name=name, description=f"demo {name}")
```

### Variante A — compose once, inject globally

This is what `prismal-server` does in its lifespan. The host either builds the full default composite (async, because MCP servers must connect) or, for the offline path, assembles the same shape by hand:

```python
async def variante_a_global() -> None:
    if os.environ.get("EXAMPLE_USE_MCP") == "1":
        # Full composite: MCP (from config/mcp_servers.yaml) + Skills + stubs.
        provider = await build_default_tool_provider()
    else:
        # Offline composite: Skills + stub fallback (same merge policy).
        provider = CompositeToolProvider([SkillToolProvider(), StubToolProvider()])

    print("providers:", [type(p).__name__ for p in provider.providers])
    set_tool_provider(provider)

    # The 26 agent nodes keep calling get_tools_for_agent unchanged.
    for agent in ("researcher", "coder", "cron_manager"):
        tools = get_tools_for_agent(agent)
        print(f"{agent:>14}: {len(tools)} tools (first: {[t.name for t in tools[:3]]})")
```

Two things are worth noting. First, `set_tool_provider()` is idempotent — the host calls it once at startup, and calling it again simply replaces the provider (last one wins). Second, `get_tools_for_agent()` keeps its historical signature: agent nodes never learned about the inversion; the facade just delegates to whatever provider was injected. If *nothing* is injected, the registry degrades to a stub-only fallback and logs a structured `tool_registry.no_provider` warning — or raises `ToolProviderNotConfigured` when `settings.tool_provider_strict` is set. Watch `cron_manager` in the output: as a member of the fixed-tool set it gets only its stubs regardless of how rich the composite is.

### Variante B — per-session providers for multi-tenant hosts

Global injection is one provider per process. When each user or tenant needs a different toolset, a real host enables `settings.tool_provider_mode = "context"` and binds the provider into the compiled graph — `get_async_compiled_graph(tool_provider=provider_for(user))` — after which nodes resolve through `get_tools_for_agent_ctx(name, config)`. The notebook exercises that resolution helper directly, with two simulated sessions running concurrently:

```python
provider_u = FakeToolProvider(default=[_make_tool("user_u_search")])
provider_v = FakeToolProvider(default=[_make_tool("user_v_search")])

config_u = {"configurable": {"tool_provider": provider_u}}
config_v = {"configurable": {"tool_provider": provider_v}}

tools_u, tools_v = await asyncio.gather(
    asyncio.to_thread(get_tools_for_agent_ctx, "researcher", config_u),
    asyncio.to_thread(get_tools_for_agent_ctx, "researcher", config_v),
)
print("session U sees:", [t.name for t in tools_u])
print("session V sees:", [t.name for t in tools_v])
```

Under the hood, `resolve_provider(config)` reads `config["configurable"]["tool_provider"]` and falls back to the global provider when the key is absent — resolution is a pair of dict lookups with no lock, so concurrent sessions are safe. Session U and session V print disjoint toolsets, demonstrating tenant isolation without any process-wide state.

Every resolution, in either variant, is observable: the registry wraps the call in a `prismal.tools.resolve` OTel span and emits counters such as `prismal.tool_provider_resolved_total{provider}`, `prismal.tools_injected_total{agent}`, and `prismal.tool_provider_fallback_total`.

### Entry point

```python
async def main() -> None:
    print("— Variante A (global) —")
    await variante_a_global()
    print("\n— Variante B (per-session) —")
    await variante_b_per_session()
```

## Key API

| Symbol | Role |
|---|---|
| `ToolProviderPort` | The hexagonal port: sync `get_tools(*, agent_name, capabilities)`, must not raise |
| `CompositeToolProvider` | Merges N providers with the legacy policy: MCP→Skills→stubs priority, name dedupe, `max_total=120`, fixed-tool agents get stubs only |
| `SkillToolProvider` / `StubToolProvider` | Adapters for active skills (fresh `SkillsManager` per call) and the static stub fallback |
| `build_default_tool_provider(settings)` | Async host-lifespan builder: MCP (cap 60) + Skills + stubs composite |
| `set_tool_provider(provider)` | Variante A — global injection point (idempotent, last one wins) |
| `get_tools_for_agent(name)` | Unchanged facade the 26 agent nodes call; delegates to the injected provider |
| `get_tools_for_agent_ctx(name, config)` | Variante B — resolves the provider from `config["configurable"]["tool_provider"]`, falling back to the global one |
| `FakeToolProvider` | Deterministic test double — no I/O, no heavy imports |

## Run it

```bash
uv run jupyter lab notebooks/tool_provider_host.ipynb
uv run python examples/tool_provider_host.py   # from the prismal repo
```

No API key is required — the demo runs offline with the Skills + stub composite. The final cell of the notebook ships with the entry point commented out (`# await main()`); uncomment it to run both variants. The call is `async`, but `nest_asyncio.apply()` in the first cell lets it run directly inside Jupyter's event loop. Set `EXAMPLE_USE_MCP=1` before launching to exercise the full MCP composite instead (requires `config/mcp_servers.yaml` in the prismal repo).

## Related

- [A custom tool provider conforming to `ToolProviderPort` (Fase Y)](tool_provider_custom.md) — the smallest possible host-supplied provider.
- [A2A tool provider](a2a_tool_provider.md) — remote agents surfaced as tools through the same port.
- [Runtime composition root (Phase R)](composition_root.md) — `build_runtime()` composes this provider together with the other core ports.
- [Supervisor quickstart](supervisor_quickstart.md) — the graph whose agent nodes consume the injected tools.
