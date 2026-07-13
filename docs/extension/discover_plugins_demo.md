# Plugin discovery with `discover_plugins()`

> **Notebook:** [`notebooks/extension/discover_plugins_demo.ipynb`](../../notebooks/extension/discover_plugins_demo.ipynb) · **Example:** [`examples/extension/discover_plugins_demo.py`](https://github.com/prismal-ai/prismal/blob/main/examples/extension/discover_plugins_demo.py)

The final piece of the Fase X extension surface is distribution: once you can build hardened nodes with `@prismal_node` and compose them with `PrismalStateGraphBuilder`, you need a way for third parties to *ship* those extensions as ordinary pip-installable packages. `discover_plugins()` provides it. A plugin author declares entry points in their `pyproject.toml` under one of four groups — `prismal.subgraphs`, `prismal.nodes`, `prismal.tools`, `prismal.rag_engines` — and after `pip install`, discovery finds and registers them automatically. No fork, no monkeypatching of framework internals, no configuration files to edit.

This demo shows the whole cycle without installing a real distribution: it fabricates an in-memory `EntryPoint`, overrides the entry-point seam that `discover_plugins()` reads, and watches a builder-compiled subgraph land in a `SubgraphRegistry`. In production the only difference is that the entry point comes from an installed package's metadata instead of a lambda.

## What it demonstrates

- Writing a plugin registration function that builds a subgraph with `PrismalStateGraphBuilder` and installs it via `SubgraphRegistry.register_sync()`.
- Simulating an installed plugin by constructing an `importlib.metadata.EntryPoint` and overriding the `plugins_mod._entry_points` test seam.
- Running `discover_plugins(settings=..., registry=..., groups=...)` and reading the resulting `DiscoveryReport` (loaded / failed / skipped counts).
- Verifying the outcome by listing the registry: the demo pipeline is now registered like any built-in subgraph.
- The isolation guarantee: each plugin loads inside its own try/except, so one broken plugin never aborts startup or the rest of discovery.

## How it works

### The plugin side: a registration function

A `prismal.subgraphs` entry point resolves to a callable that receives the registry. It can either self-register (as here, via the sync `register_sync()`) or simply return a `SubgraphDefinition` for the discoverer to register on its behalf:

```python
async def _node(state: dict) -> dict:
    return {"metadata": {"demo": True}}


def register_demo_pipeline(registry: SubgraphRegistry) -> None:
    builder = PrismalStateGraphBuilder("demo_plugin_pipeline")
    builder.add_node("n", _node)
    builder.set_entry_point("n")
    builder.add_edge("n", "__end__")
    registry.register_sync("demo_plugin_pipeline", builder.compile())
```

Note that `builder.compile()` — not `compile_raw()` — is the right call here: it returns the declarative `SubgraphDefinition` the registry stores, while `compile_raw()` would produce an already-compiled graph meant for direct invocation. In a real plugin, this function would live in your package and be pointed at from your `pyproject.toml`:

```toml
[project.entry-points."prismal.subgraphs"]
demo_plugin_pipeline = "my_pkg:register_demo_pipeline"
```

### The host side: faking an installed distribution

To run without installing anything, the demo constructs the same `EntryPoint` object that `importlib.metadata` would yield for that TOML declaration, and swaps it in through the module's test seam:

```python
ep = EntryPoint(
    name="demo_plugin_pipeline",
    value=f"{__name__}:register_demo_pipeline",
    group="prismal.subgraphs",
)
plugins_mod._entry_points = lambda group: [ep] if group == "prismal.subgraphs" else []
```

`_entry_points` is the single function through which discovery reads installed metadata, deliberately overridable so examples and tests can inject synthetic plugins.

### Discovery and the report

```python
registry = SubgraphRegistry()
report = discover_plugins(
    settings=Settings(plugins_autodiscover=True),
    registry=registry,
    groups=["subgraphs"],
)
print(f"loaded={report.loaded_count} failed={report.failed_count}")
print("registered subgraphs:", registry.list())
```

`discover_plugins()` is synchronous and walks each requested group (all four enabled groups when `groups` is omitted and `plugins_autodiscover` is on). For every entry point it:

1. Checks `settings.plugins_denylist` and `plugins_allowlist` (denylist wins; a non-empty allowlist excludes everything not on it) — filtered plugins are reported as skipped, never loaded.
2. Loads and installs the plugin with the group-specific loader — subgraphs register in the `SubgraphRegistry`; `prismal.nodes` must already carry `@prismal_node` (importing them registers their metadata); `prismal.tools` land in a plugin tool registry that respects the global 120-tool cap; `prismal.rag_engines` register in the `RAGEngineRegistry` singleton.
3. Isolates failures: any exception is caught, wrapped as a `PluginLoadError`, logged, and audited — the loop continues. A broken third-party plugin cannot take the host down at startup.

Every load, failure, and skip lands in the returned `DiscoveryReport` with per-plugin timing, and each outcome is also recorded by the `AuditLogger` — so an operator can always answer "what third-party code did this process load, and did any of it fail?". The expected output here is `loaded=1 failed=0` and a registry listing containing `demo_plugin_pipeline`.

Unlike the other extension notebooks, `main()` here is a plain synchronous function — discovery does no async work.

### Inspecting plugins from the command line

Discovery has a CLI counterpart for operations work. `list_plugins()` and `get_plugin_info()` inspect installed entry points *without* loading them, and the same functionality ships as a module CLI:

```bash
python -m prismal.plugins list     # every installed plugin across enabled groups
python -m prismal.plugins info <name>
python -m prismal.plugins doctor   # attempt loads and report failures
```

Two conventions round out the plugin story. Distribution naming: publish plugins as `prismal-x-<domain>` so they are easy to find on PyPI. And scope: the four groups map one-to-one onto the extension surface — subgraphs are `SubgraphDefinition`s (usually built with `PrismalStateGraphBuilder`), nodes are `@prismal_node` callables, tools are `BaseTool` instances (or zero-arg factories) that flow to agents through the injected tool provider, and RAG engines are classes implementing the RAG engine protocol.

## Key API

| Symbol | Role |
|---|---|
| `discover_plugins(settings=, registry=, groups=)` | Discovers entry points across the four groups, applies allowlist/denylist, loads each plugin in isolation. |
| `DiscoveryReport` | Aggregate result: `loaded` / `failed` / `skipped` lists plus counts and total duration. |
| `PluginInfo` / `PluginLoadResult` | Per-plugin static metadata and per-load outcome (status, error, duration). |
| `SubgraphRegistry.register_sync(name, definition)` | Sync registration path plugins use to install a `SubgraphDefinition`. |
| `PrismalStateGraphBuilder.compile()` | Produces the registrable `SubgraphDefinition` a subgraph plugin contributes. |
| `plugins_mod._entry_points` | Overridable seam that reads `importlib.metadata.entry_points()` — how this demo injects a fake plugin. |
| `Settings.plugins_allowlist` / `plugins_denylist` | Operator-controlled filters on which plugins may load (denylist wins). |
| `RAGEngineRegistry` | Where `prismal.rag_engines` plugins are registered. |
| `list_plugins()` / `get_plugin_info()` | Inspect installed plugins without loading them (also via `python -m prismal.plugins`). |

## Run it

```bash
uv run jupyter lab notebooks/extension/discover_plugins_demo.ipynb
uv run python examples/extension/discover_plugins_demo.py   # from the prismal repo
```

Runs fully offline against an in-memory entry point — no plugin installation and no API keys required.

## Related

- [PrismalStateGraphBuilder](custom_subgraph.md) — building the subgraph a plugin contributes.
- [The `@prismal_node` decorator](custom_node.md) — the contract `prismal.nodes` plugins must satisfy.
- [Composition root](../composition_root.md) — where a host assembles ports and settings at startup.
- [Host tool providers](../tool_provider_host.md) — how contributed tools reach agents through the tool-provider port.
