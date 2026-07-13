# Injecting the default environment config source (Fase W)

> **Notebook:** [`notebooks/config_source_env.ipynb`](../notebooks/config_source_env.ipynb) · **Example:** [`examples/config_source_env.py`](https://github.com/prismal-ai/prismal/blob/main/examples/config_source_env.py)

Fase W inverts configuration as a hexagonal port, following the same playbook as Fase Y (tools) and Fase Z (vector stores). The core no longer *reads* its own environment — no `.env` discovery, no `os.environ` scanning scattered through modules. Instead it *consumes* an injected `ConfigSourcePort`: a `@runtime_checkable` Protocol with a single sync `load()` that returns a `Mapping[str, str | SecretStr]` and must never raise. `Settings` keeps its full Pydantic schema and does what it always did — validate — while *where the raw strings come from* becomes the host's decision.

This notebook shows the simplest, single-tenant host path: build the default adapter, `EnvConfigSource`, and inject it once at startup with `set_config_source()`. The inversion is deliberately invisible here — `EnvConfigSource` reproduces the pre-Fase-W behaviour byte-for-byte, and a deployment that injects nothing at all falls back to the very same source. The payoff comes later, when the same one-line injection point accepts a Vault client or a tenant row instead (see the [custom sources notebook](config_source_custom.md)).

## What it demonstrates

- The `ConfigSourcePort` contract: one sync `load()` returning a raw config mapping, never raising.
- `EnvConfigSource` as the default adapter — the *only* place in the core that reads `os.environ` and the `.env` file.
- One-time global injection with `set_config_source()`, which also invalidates the `get_settings()` cache.
- That every subsequent `get_settings()` call across the framework resolves through the injected source — validated fields come out the other end.
- Backward compatibility: no injection at all yields identical behaviour, so existing deployments need zero changes.

## How it works

### Imports: the port's default adapter and the injection registry

Two imports carry the whole demo — the settings accessor the entire framework already uses, and the Fase W source machinery.

```python
from prismal.core.config import get_settings
from prismal.core.config_source import EnvConfigSource, set_config_source
```

`prismal.core.config_source` is the home of the port. Besides `EnvConfigSource` it ships `MappingConfigSource` (an in-memory dict source), `ChainedConfigSource` (an ordered first-wins composite that skips a failing sub-source), and `FakeConfigSource` (a pure test double) — the standard Protocol + adapters + fake + injection-registry set every prismal port follows.

### The port contract

The Protocol itself is minimal by design — one method, sync, infallible (from `prismal/core/config_source.py`):

```python
@runtime_checkable
class ConfigSourcePort(Protocol):
    """A source of raw configuration values consumed by ``Settings``."""

    def load(self) -> Mapping[str, ConfigValue]:
        """Return the raw configuration mapping. Sync; must not raise."""
        ...
```

`ConfigValue` is `str | SecretStr`. The core only ever invokes `load()`; it never constructs concrete sources, and conformance is structural (`@runtime_checkable`), so a host source needs no import of a base class. Keeping `load()` synchronous and non-raising is what makes the whole scheme composable: `Settings` construction never has to handle a source failure mid-validation, and a chained composite can treat a misbehaving sub-source as absent rather than fatal.

### Startup: inject the source once

The host owns configuration loading, so it performs the injection exactly once, at startup:

```python
def on_startup() -> None:
    """Own configuration loading: inject the env source once at startup."""
    set_config_source(EnvConfigSource(dotenv_path=".env"))
```

`EnvConfigSource.load()` resolves three layers into one mapping, highest precedence first:

1. the live environment — `PRISMAL_*` variables plus a small allowlist of unprefixed provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, the Langfuse trio) kept for parity with historical behaviour;
2. the dotenv file at `dotenv_path` (default `".env"`; a missing file is not an error);
3. the legacy `LIGHTAGENT_* → PRISMAL_*` mirror, folded in only where the `PRISMAL_` name is unset, with a single `DeprecationWarning` per process.

The mirror detail matters: it is computed *into the returned mapping*, never by mutating `os.environ`. The old `env_compat` module used to rewrite the process environment at import time; Fase W removed that global side effect entirely — importing `prismal.core` now mutates zero environment variables.

`set_config_source()` stores the source in a global registry and calls `reload_settings()`, clearing the `get_settings()` `@lru_cache` so the very next read rebuilds `Settings` through the new source. Passing `None` clears the injection and the default resumes.

### After injection: settings resolve through the port

From here on, nothing in the calling code changes — the roughly 151 `get_settings()` call sites across the framework are untouched by Fase W:

```python
def main() -> None:
    on_startup()
    settings = get_settings()
    print("default_model:", settings.default_model)
    print("vector_store_backend:", settings.vector_store_backend)
    print("config_source_strict:", settings.config_source_strict)
```

Internally, `get_settings()` delegates to `build_settings()`, which resolves the active source (an explicit argument, the injected global, or the default `EnvConfigSource`) and constructs `Settings` with the port plugged in as its Pydantic settings source. The adapter — a `_ConfigSourceSettingsSource` subclassing `EnvSettingsSource` — replaces only the raw key/value loading, so prefix handling, `AliasChoices` resolution, case-insensitivity, and JSON decoding of complex fields (`list[str]`, …) all keep working exactly as before. Validation stays entirely in `Settings`; the source supplies strings, nothing more.

The printed `config_source_strict` field hints at the hardened variant: with `PRISMAL_CONFIG_SOURCE_STRICT=true`, a host that *must never* read ambient environment gets a `ConfigSourceError` if no source was injected, instead of the silent `EnvConfigSource` fallback.

### Precedence at a glance

For a given key, `EnvConfigSource` resolves the winning value in this order (highest first):

| Priority | Layer | Example |
|---|---|---|
| 1 | Live environment, `PRISMAL_*` | `PRISMAL_DEFAULT_MODEL=claude-sonnet-4-5` |
| 2 | Live environment, allowlisted unprefixed keys | `ANTHROPIC_API_KEY=sk-...` |
| 3 | `.env` file at `dotenv_path` | `PRISMAL_VECTOR_STORE_BACKEND=chroma` |
| 4 | Legacy mirror, `LIGHTAGENT_*` (deprecated) | `LIGHTAGENT_DEFAULT_MODEL` → `PRISMAL_DEFAULT_MODEL` |

Field defaults declared on `Settings` apply only when no layer supplies the key at all — defaults are validation-time behaviour, not part of the source.

### Why this is byte-for-byte compatible

The design goal was that doing nothing is safe. If a deployment never calls `set_config_source()`, `build_settings()` constructs a default `EnvConfigSource()` on the fly — same `.env` discovery, same `PRISMAL_` prefix, same unprefixed provider keys, same legacy mirror. The demo's explicit injection produces an identical mapping; its value is making the ownership explicit and giving the host a single seam to replace later.

### What this buys tests and multi-tenant hosts

The same seam serves two audiences beyond the startup path. Tests inject a `FakeConfigSource` (or an `EnvConfigSource(dotenv_path=None)` that ignores any stray `.env` on disk — this is exactly what prismal's own autouse test fixture does), getting deterministic settings with zero environment coupling. Multi-tenant hosts skip the global registry entirely and call `build_settings(source)` per tenant: it activates the source through a `ContextVar` for the duration of the construction, so concurrent per-tenant builds never interfere — the pattern the [custom sources notebook](config_source_custom.md) develops in full.

An architectural guard keeps the inversion honest: an AST test in the prismal repo (`tests/unit/core/test_no_env_reads.py`) forbids new literal `os.getenv` / `os.environ` configuration reads anywhere in the core, with `core/config_source.py` itself as the primary exemption. Configuration reads that used to be scattered — the Tavily key in the tools module, MCP token lookups — were relocated behind `Settings` fields or source-aware resolvers as part of the same phase.

## Key API

| Symbol | Role |
|---|---|
| `ConfigSourcePort` | `@runtime_checkable` Protocol — sync `load() -> Mapping[str, str \| SecretStr]`, must not raise |
| `EnvConfigSource(dotenv_path=".env", env=None, include_legacy_aliases=True)` | Default adapter; the only core reader of `os.environ`/`.env`; folds in the `LIGHTAGENT_` mirror without global mutation |
| `set_config_source(source)` | Injects (or clears, with `None`) the global source; invalidates the `get_settings()` cache via `reload_settings()` |
| `get_config_source()` | Returns the currently injected source, or `None` |
| `get_settings()` | Cached (`@lru_cache`) settings accessor — unchanged signature, now resolves through the port |
| `build_settings(source=None)` | Pure, uncached constructor over a source — the per-tenant / per-test path |
| `reload_settings()` | Clears the `get_settings()` cache so the next call rebuilds from the active source |
| `Settings.config_source_strict` | When `True`, a missing injected source raises `ConfigSourceError` instead of falling back to env |
| `FakeConfigSource(values)` | Deterministic test double — no environment access, no `.env` discovery |
| `ConfigValue` | The value union a source may return: `str \| SecretStr` |

## Run it

```bash
uv run jupyter lab notebooks/config_source_env.ipynb
uv run python examples/config_source_env.py   # from the prismal repo
```

No API key is required — the demo runs offline (it only reads whatever environment happens to be present and prints validated defaults). In the notebook, the final run cell ships as the commented `# main()`; uncomment it to execute. `main()` is a plain synchronous function, so no `await` is needed.

## Related

- [Custom config sources and per-tenant settings (Fase W)](config_source_custom.md) — replacing `EnvConfigSource` with a Vault-style source and building isolated per-tenant `Settings`.
- [Runtime composition root (Fase R)](composition_root.md) — the facade that consumes these settings to assemble every core port.
- [Host tool provider (Fase Y)](tool_provider_host.md) — the sibling port inversion this phase mirrors.
- [Runtime hardening](runtime_hardening.md) — further opt-in strictness layers for production hosts.
