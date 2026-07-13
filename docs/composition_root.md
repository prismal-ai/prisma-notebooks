# Runtime composition root: one facade for every port (Fase R)

> **Notebook:** [`notebooks/composition_root.ipynb`](../notebooks/composition_root.ipynb) · **Example:** [`examples/composition_root.py`](https://github.com/prismal-ai/prismal/blob/main/examples/composition_root.py)

Fase R closes the loop on prismal's hexagonal architecture. Earlier phases inverted individual dependencies as ports — tools became `ToolProviderPort` (Fase Y), vector search became `VectorStorePort` (Fase Z) — but that left the host to assemble each port by hand, in the right order, with the right teardown. The composition root is the single facade that does the assembling: one call to `build_runtime()` takes `settings` plus an optional tenant (`org_id`) and returns a `RuntimeContext` grouping five composed ports — tool provider, vector store provider, embeddings, checkpointer, and audit logger.

Its guiding principle is **orchestrate, do not reimplement**: `build_runtime()` calls the existing builders (`build_default_tool_provider` from Fase Y, `VectorStoreFactory` from Fase Z, `EmbeddingsFactory`, `build_checkpointer`, `AuditLogger`) and never duplicates their logic. Because the real facade connects MCP servers, opens a checkpointer, and loads an embeddings model, this notebook runs the deterministic sibling `build_test_runtime()` instead and only *describes* the host lifespan a server would run.

## What it demonstrates

- Composing a complete `RuntimeContext` with fakes via `build_test_runtime()` — five ports, zero I/O, no global state.
- The `RuntimeContext` as an async context manager with an idempotent `aclose()` that releases everything the composition created (MCP connections, checkpointer, built vector stores).
- Per-tenant isolation with `collection_for(base, org_id)` — the same suffix rule applied identically to RAG and memory collections.
- The two runtime modes: `global` (injects the tool-provider singleton) versus `context` (every port stays inside the returned context).
- The real host lifespan pattern: `build_runtime(get_settings())` once at startup, `ctx.aclose()` at shutdown.

## How it works

### Imports: the composition facade and its fakes

The example needs only three things: the test facade, the tenant-naming helper, and two fake ports to inject.

```python
from prismal.agents.extension.providers import FakeToolProvider
from prismal.composition import build_test_runtime, collection_for
from prismal.rag.vector_store_factory import FakeVectorStore
```

`FakeToolProvider` is the Fase Y test double (a `ToolProviderPort` backed by a plain dict), and `FakeVectorStore` is the Fase Z deterministic, I/O-free store. Both come from the same modules production code uses — the fakes are first-class citizens of each port, not test-only bolt-ons.

### Step 1 — a test runtime: five ports, no I/O

`build_test_runtime()` mirrors `build_runtime()`'s output shape but wires deterministic doubles for anything not supplied: a `FakeToolProvider`, a `FakeVectorStore`, and in-module embeddings/checkpointer/audit fakes. It performs no I/O, opens no connections, and injects nothing globally.

```python
async def demo_test_runtime() -> None:
    """1. Compose a runtime with fakes — five ports, no I/O."""
    ctx = build_test_runtime(
        tool_provider=FakeToolProvider({}),
        vector_store=FakeVectorStore(),
        org_id="acme",
    )
    print("ports:", ctx.tool_provider, ctx.embeddings, ctx.checkpointer, ctx.audit)
    print("collection:", ctx.config.collection_name)  # default_acme
    async with ctx:  # aclose() is a no-op for the test runtime
        pass
```

The returned `RuntimeContext` is a pure container — it adds no business behaviour; consumers keep talking to the ports directly. Its `config` attribute is a frozen `RuntimeConfig` snapshot (`org_id`, `runtime_mode`, backend, collection names) with sensitive values deliberately left in `settings`, not copied into the config view. Because `org_id="acme"` was passed, the resolved collection name is `default_acme`: the base collection `"default"` suffixed by the tenant.

Note one Fase R subtlety visible here: the context carries a `vector_store_provider` (a `VectorStoreProvider` wrapping `VectorStoreFactory`), not a bare store. Fase Z ships a *factory*, not a process singleton, so the vector store is always resolved through the context — `ctx.vector_store_provider.get_store(...)` — and every collection it builds is automatically tenant-suffixed. There is no `set_vector_store_provider` global, even in `global` mode.

### Step 2 — the host lifespan (what `prismal-server` runs)

The runnable demo stays offline, so the production path is shown as a documented string rather than executed. It is the canonical usage: compose once in the app lifespan, tear down once at shutdown.

```python
from contextlib import asynccontextmanager
from prismal.agents.graph import get_async_compiled_graph
from prismal.composition import build_runtime
from prismal.core.config import get_settings

@asynccontextmanager
async def lifespan(app):
    ctx = await build_runtime(get_settings())          # mode=global
    app.state.runtime = ctx
    app.state.graph = await get_async_compiled_graph()
    try:
        yield
    finally:
        await ctx.aclose()
```

`build_runtime(settings=None, *, org_id=None, overrides=None, mode=None, collection_base="default", mcp_config_path=None)` composes the ports in order — tool provider (the only async sub-builder, since it connects MCP), vector store provider, embeddings, checkpointer, audit — registering a teardown closer for each resource it opens. If any sub-port fails, it runs the closers accumulated so far and raises `RuntimeCompositionError(port, cause)`, so a half-built runtime never leaks connections.

The `mode` decides where the tool provider lives. In `global` mode (the default single-tenant path) `build_runtime` finishes by calling `set_tool_provider()`, so the compiled graph picks the provider up with no extra wiring. In `context` mode nothing is injected: every port stays inside the `RuntimeContext`, which is what multi-tenant hosts use to run several runtimes side by side. When `mode` is not passed, it is derived from `settings.runtime_mode`. `aclose()` is idempotent — closers run once, in reverse (LIFO) order, and never raise on already-closed resources.

### Failure semantics: no half-built runtimes

Composition can fail at any of the five steps — an unreachable MCP server, a missing vector-store extra, a locked checkpointer database. Each sub-builder is wrapped so that a failure surfaces as a single, attributable exception:

```python
try:
    tool_provider = await build_default_tool_provider(
        eff, mcp_config_path=mcp_config_path
    )
except Exception as exc:
    raise RuntimeCompositionError("tool_provider", str(exc)) from exc
```

Before re-raising, `build_runtime` runs every teardown closer registered up to that point (MCP disconnect, vector-store release, checkpointer close), so a failure at step 4 does not leak the connections opened in steps 1–3. The `port` attribute on `RuntimeCompositionError` names the failing step (`"tool_provider"`, `"vector_store"`, `"embeddings"`, `"checkpointer"`, `"audit"`), which makes startup failures diagnosable from the exception alone. The same closer list becomes the context's `aclose()` on success — one teardown path for both outcomes.

### Step 3 — tenant isolation is one pure function

Tenant resolution is deliberately boring: `collection_for(base, org_id)` returns `base` unchanged for the single-tenant case and `f"{base}_{org_id}"` otherwise.

```python
def demo_tenant_isolation() -> None:
    """3. The same base collection is isolated per tenant."""
    base = "docs"
    print("single tenant:", collection_for(base, None))  # docs
    print("acme        :", collection_for(base, "acme"))  # docs_acme
    print("globex      :", collection_for(base, "globex"))  # docs_globex
```

Because the *same* rule is applied to RAG and memory collections, a tenant sees one consistent, isolated namespace across both subsystems — two `build_runtime(..., org_id=...)` calls for `acme` and `globex` can run in the same process without ever touching each other's vectors. Per-tenant setting tweaks travel through the `overrides` parameter (applied as a validated `model_copy`, leaving the global settings untouched), and a fully isolated per-tenant configuration can be threaded in via `apply_org_overrides(..., source=...)` from Fase W.

Multi-tenant hosting therefore composes naturally: run each tenant's runtime in `context` mode so nothing is injected globally, keep the contexts keyed by `org_id`, and let `collection_for` guarantee that their vector data never collides. Single-tenant hosts get the simpler `global` mode and one lifespan-scoped context, as shown in step 2.

## Key API

| Symbol | Role |
|---|---|
| `build_runtime(settings, *, org_id, overrides, mode, collection_base, mcp_config_path)` | Async facade — composes all five ports from settings + optional tenant; raises `RuntimeCompositionError` and tears down partial state on failure |
| `build_test_runtime(*, tool_provider, vector_store, embeddings, org_id, ...)` | Deterministic sibling — fakes for every unsupplied port, no I/O, no-op `aclose()` |
| `RuntimeContext` | Groups the composed ports + `org_id`; async context manager with idempotent, LIFO `aclose()` |
| `RuntimeConfig` | Frozen resolved view of one runtime (mode, backend, collection names); secrets stay in `settings` |
| `VectorStoreProvider.get_store(collection_name=None)` | Builds tenant-suffixed `VectorStorePort` instances via `VectorStoreFactory` (Fase Z is factory-based) |
| `collection_for(base, org_id)` | Pure tenant naming — `"docs"` → `"docs_acme"`; applied identically to RAG and memory |
| `RuntimeCompositionError(port, cause)` | Raised when a sub-port fails to compose, after partial teardown |
| `FakeToolProvider` / `FakeVectorStore` | The Fase Y / Fase Z test doubles injected in this demo |

## Run it

```bash
uv run jupyter lab notebooks/composition_root.ipynb
uv run python examples/composition_root.py   # from the prismal repo
```

No API key is required — the demo runs entirely offline with injected fakes. In the notebook, the final run cell ships as the commented `# main()`; uncomment it to execute. `main()` is synchronous and calls `asyncio.run(...)` internally — the `nest_asyncio.apply()` in the first cell makes that safe inside Jupyter's already-running event loop.

## Related

- [Inject the default environment config source (Fase W)](config_source_env.md) — the configuration port that feeds `settings` into this facade.
- [Custom config sources and per-tenant settings (Fase W)](config_source_custom.md) — per-tenant `Settings` that pair with `build_runtime(org_id=...)`.
- [Host tool provider (Fase Y)](tool_provider_host.md) — the `build_default_tool_provider` path the composition root reuses.
- [LanceDB vector store (Fase Z)](vector_store_lancedb.md) — swapping the backend the `VectorStoreProvider` builds.
