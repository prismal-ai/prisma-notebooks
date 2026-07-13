# A custom tool provider conforming to `ToolProviderPort` (Fase Y)

> **Notebook:** [`notebooks/tool_provider_custom.ipynb`](../notebooks/tool_provider_custom.ipynb) · **Example:** [`examples/tool_provider_custom.py`](https://github.com/prismal-ai/prismal/blob/main/examples/tool_provider_custom.py)

Fase Y turns tool resolution into a hexagonal port: instead of the agent core importing `prismal.mcp` and `prismal.skills` to fetch tools (a dependency now forbidden by an AST architecture test), the core asks an injected `ToolProviderPort` — a sync `get_tools(*, agent_name, capabilities) -> list[BaseTool]` that must never raise. The framework ships adapters for the standard sources (MCP, skills, stubs, and a composite of all three), but the port is deliberately open: *any* object with the right method shape qualifies.

This notebook demonstrates exactly that openness. It builds the smallest possible host-supplied provider — a plain Python class with one method, no base class, no inheritance, no registration ceremony — verifies it structurally conforms to the port, injects it once with `set_tool_provider()`, and shows the unchanged `get_tools_for_agent()` facade resolving tools through it. This is the pattern for wiring a company-internal tool catalog, a database-backed registry, or a feature-flag service into prismal's 26 agent nodes.

## What it demonstrates

- Writing a provider as a plain class: the only requirement is `get_tools(*, agent_name, capabilities=None) -> list[BaseTool]`.
- Structural (duck-typed) conformance: `ToolProviderPort` is a `@runtime_checkable` Protocol, checked with the `conforms_to()` helper — no subclassing needed.
- The port's error contract: a degraded source returns `[]` instead of raising, so tool resolution can never crash an agent turn.
- Variante A injection: one `set_tool_provider(provider)` call at startup, after which every agent node resolves through the custom provider via the unchanged `get_tools_for_agent()` facade.
- Per-agent tool scoping: `researcher` and `coder` get their catalog entries; an unknown agent (`planner`) cleanly gets an empty list.

## How it works

### Imports: two symbols from the extension surface, two from the registry

```python
from langchain_core.tools import StructuredTool

from prismal.agents.extension import ToolProviderPort, conforms_to
from prismal.agents.tool_registry import get_tools_for_agent, set_tool_provider
```

`ToolProviderPort` is the Protocol itself (re-exported from `prismal/agents/extension/__init__.py`, defined in `agents/extension/ports.py` alongside the other hexagonal ports). `conforms_to(obj, port)` wraps `isinstance` against a `@runtime_checkable` Protocol and returns `False` rather than raising for non-conforming objects — a convenient startup assertion. Tools themselves are ordinary LangChain `BaseTool` instances; the demo builds them from plain functions:

```python
def _make_tool(name: str, description: str) -> StructuredTool:
    def _fn(query: str = "") -> str:
        return f"[{name}] {query}"

    return StructuredTool.from_function(func=_fn, name=name, description=description)
```

### The provider: a dict lookup behind the port contract

The whole provider is a class holding a per-agent catalog. Replace the dict with a database query, an HTTP call to an internal service, or a feature-flag check — the shape stays the same:

```python
class MyToolProvider:
    """Per-agent lookup backed by a plain dict — replace with your own source.

    Conforms ``ToolProviderPort`` structurally: the only requirement is
    ``get_tools(*, agent_name, capabilities=None) -> list[BaseTool]``.
    It must not raise on a degraded source — return ``[]`` instead.
    """

    def __init__(self) -> None:
        self._catalog = {
            "researcher": [
                _make_tool("acme_search", "Search the ACME internal knowledge base"),
            ],
            "coder": [
                _make_tool("acme_ci_status", "Check the ACME CI pipeline status"),
                _make_tool("acme_deploy", "Trigger an ACME staging deploy"),
            ],
        }

    def get_tools(
        self,
        *,
        agent_name: str,
        capabilities: list[str] | None = None,
    ) -> list[StructuredTool]:
        del capabilities  # this provider does not filter by capability
        return self._catalog.get(agent_name, [])
```

Three aspects of the contract matter here. The parameters are **keyword-only** (`*, agent_name, capabilities`), matching the Protocol's signature. The method is **sync** — the core calls it on the hot path of a graph node, so any slow I/O belongs in a prepare step at startup, not inside `get_tools`. And it **must not raise**: if the backing source is down, return `[]` and let the agent proceed tool-less, exactly as the built-in adapters do. The `capabilities` argument is the Fase E filter (`DEFAULT_CAPABILITY_MAP` recommendations); a provider is free to ignore it, as this one does.

### Injection and resolution

```python
def main() -> None:
    provider = MyToolProvider()

    # Structural conformance — no inheritance needed.
    print("conforms to ToolProviderPort:", conforms_to(provider, ToolProviderPort))

    # Variante A: inject once at startup; every agent node resolves through it.
    set_tool_provider(provider)

    for agent in ("researcher", "coder", "planner"):
        tools = get_tools_for_agent(agent)
        print(f"{agent:>10}: {[t.name for t in tools]}")
```

`conforms_to` prints `True`: `MyToolProvider` satisfies the Protocol purely by shape. `set_tool_provider(provider)` is variante A — the global injection point a host calls once in its lifespan (calling it again just replaces the provider). From then on the unchanged facade `get_tools_for_agent()` delegates every lookup to the custom provider, wrapped in the `prismal.tools.resolve` OTel span and the `prismal.tool_provider_resolved_total` / `prismal.tools_injected_total` counters.

The output shows the scoping: `researcher` gets `['acme_search']`, `coder` gets `['acme_ci_status', 'acme_deploy']`, and `planner` — absent from the catalog — gets `[]`. Contrast that with the *no provider at all* case: without any injection, the registry falls back to the static stubs with a `tool_registry.no_provider` warning (or raises `ToolProviderNotConfigured` when `settings.tool_provider_strict` is on). A custom provider that intentionally returns an empty list is a normal, silent outcome.

For multi-tenant hosts that need a *different* custom provider per session, the same class works unchanged with variante B — passed to `get_async_compiled_graph(tool_provider=...)` under `tool_provider_mode="context"` — as shown in the host composition notebook. And nothing stops you from composing it with the standard sources: `CompositeToolProvider([my_provider, SkillToolProvider(), StubToolProvider()])` merges it under the framework's dedupe and cap policy.

## Key API

| Symbol | Role |
|---|---|
| `ToolProviderPort` | `@runtime_checkable` Protocol: sync `get_tools(*, agent_name, capabilities)`, must not raise |
| `conforms_to(obj, port)` | Structural conformance check against a runtime-checkable Protocol (never raises) |
| `MyToolProvider` | The host-supplied implementation — a plain class, no base class or registration |
| `set_tool_provider(provider)` | Variante A global injection point; idempotent, last one wins |
| `get_tools_for_agent(name)` | The stable facade agent nodes call; delegates to the injected provider |
| `StructuredTool.from_function` | LangChain helper turning a plain function into a `BaseTool` |

## Run it

```bash
uv run jupyter lab notebooks/tool_provider_custom.ipynb
uv run python examples/tool_provider_custom.py   # from the prismal repo
```

No API key is required — the demo runs entirely offline. The notebook's final cell ships with the sync entry point commented out (`# main()`); uncomment and run it. Since `main` is synchronous, no `await` or event-loop handling is involved.

## Related

- [Host-style tool provider composition and injection (Fase Y)](tool_provider_host.md) — composing the standard adapters and the per-session variante B.
- [A2A tool provider](a2a_tool_provider.md) — remote A2A agents exposed through the same `ToolProviderPort`.
- [Runtime composition root (Phase R)](composition_root.md) — how `build_runtime()` assembles the tool provider with the other ports.
- [Swapping the vector store backend by configuration (Fase Z)](vector_store_lancedb.md) — the mirror inversion for vector search.
