# Custom config sources: a Vault-style source and per-tenant settings (Fase W)

> **Notebook:** [`notebooks/config_source_custom.ipynb`](../notebooks/config_source_custom.ipynb) · **Example:** [`examples/config_source_custom.py`](https://github.com/prismal-ai/prismal/blob/main/examples/config_source_custom.py)

Fase W turned configuration into a hexagonal port: the core consumes an injected `ConfigSourcePort` — a Protocol with one sync `load()` returning a `Mapping[str, str | SecretStr]` that must never raise — while `Settings` keeps its schema and only validates. The [companion notebook](config_source_env.md) injects the default `EnvConfigSource` and changes nothing observable; this one shows why the inversion was worth it. Because the port is structural (`@runtime_checkable`, no base class to inherit), *any* plain class with a conforming `load()` is a valid source — a Vault client, an AWS Secrets Manager wrapper, a database row.

The notebook demonstrates the two host patterns that fall out of that: a **secrets-manager source** chained ahead of the environment so vault values win globally, and **per-tenant settings** built from an in-memory mapping through `build_settings()` with no global state at all — parallel tenants never share configuration.

## What it demonstrates

- Writing a custom source: a plain class with a sync `load()` — no inheritance, structural conformance to `ConfigSourcePort`.
- Wrapping secrets in `SecretStr` so they never leak into logs or `repr`.
- `ChainedConfigSource` precedence: first source wins, and a failing sub-source is logged and skipped rather than propagated.
- Global injection with `set_config_source()` versus per-call construction with `build_settings(source)`.
- Tenant isolation: two `MappingConfigSource`-backed `Settings` objects with different models, coexisting in one process.

## How it works

### Imports: everything comes from two modules

The port machinery lives in `prismal.core.config_source`; the pure per-tenant constructor lives next to `Settings` in `prismal.core.config`:

```python
from pydantic import SecretStr

from prismal.core.config import build_settings
from prismal.core.config_source import (
    ChainedConfigSource,
    ConfigValue,
    EnvConfigSource,
    MappingConfigSource,
    set_config_source,
)
```

Note what is *not* imported: no base class for the custom source. `ConfigValue` is imported only as a type annotation for `load()`'s return mapping.

### Pattern 1 — a Vault-style config source

The whole "custom adapter" is a few lines. There is nothing to subclass and nothing to register — conforming to the port means having the right `load()` shape:

```python
class VaultConfigSource:
    """Toy secrets-manager source — replace ``_read`` with a real vault client.

    Conforms ``ConfigSourcePort`` structurally: a sync ``load()`` that never
    raises and returns a mapping. Secrets are wrapped in ``SecretStr`` so they
    never leak into logs or ``repr``.
    """

    def __init__(self, secrets: Mapping[str, str]) -> None:
        self._secrets = dict(secrets)

    def load(self) -> Mapping[str, ConfigValue]:
        return {key: SecretStr(value) for key, value in self._secrets.items()}
```

Two contract points deserve attention. First, `load()` must not raise — a production version would catch vault timeouts internally and return what it has (or an empty mapping). Second, values may be `str` or `SecretStr` (`ConfigValue` is the union); returning `SecretStr` means the raw secret is masked everywhere until someone explicitly calls `.get_secret_value()`.

At startup the host chains the vault source *ahead of* the environment source and injects the composite globally:

```python
def secrets_manager_startup() -> None:
    """Vault secrets override env/file defaults (vault first in the chain)."""
    set_config_source(
        ChainedConfigSource(
            [
                VaultConfigSource({"PRISMAL_ANTHROPIC_API_KEY": "sk-from-vault"}),
                EnvConfigSource(dotenv_path=".env"),
            ]
        )
    )
```

`ChainedConfigSource` merges its sub-sources in order with first-wins semantics: the vault's `PRISMAL_ANTHROPIC_API_KEY` shadows any value from the environment or `.env`, while every key the vault does not supply falls through to `EnvConfigSource` — so the rest of the configuration keeps behaving exactly as before. If a sub-source's `load()` raises despite the contract, the chain logs `config_source.subsource_error` and skips it (the same resilience policy as Fase Y's `CompositeToolProvider`). `set_config_source()` also invalidates the `get_settings()` cache, so the very next read resolves through the new chain.

### Pattern 2 — per-tenant settings, no globals

Global injection is right for a single-tenant host, but a multi-tenant server cannot let tenants race over one process-wide source. The second pattern skips the registry entirely: each tenant's config row becomes a `MappingConfigSource`, and `build_settings(source)` constructs an isolated, validated `Settings` from it.

```python
def tenant_settings(org_id: str):
    """Build isolated Settings for one tenant from its config row (no globals)."""
    tenant_row = {
        "acme": {"PRISMAL_DEFAULT_MODEL": "claude-opus-4-8"},
        "globex": {"PRISMAL_DEFAULT_MODEL": "gpt-4o"},
    }[org_id]
    return build_settings(MappingConfigSource(tenant_row))
```

`build_settings()` is a pure constructor over a source — no caching, no global state. Internally it activates the source through a `ContextVar` for the duration of the `Settings` construction, so concurrent `build_settings(source)` calls for different tenants never clobber each other. The `acme` object reports `claude-opus-4-8`, the `globex` object reports `gpt-4o`, and neither affects `get_settings()` or the other tenant. This is exactly the seam the composition root builds on: `apply_org_overrides(settings, org_id, overrides, source=...)` threads a per-tenant source through `build_settings()` when assembling a tenant's runtime.

### Putting both together

The entry point runs the global pattern first, then the per-tenant one:

```python
def main() -> None:
    secrets_manager_startup()
    from prismal.core.config import get_settings

    key = get_settings().anthropic_api_key.get_secret_value()
    print("anthropic key (from vault):", key)

    for org in ("acme", "globex"):
        print(f"{org} default_model:", tenant_settings(org).default_model)
```

`get_settings()` — the cached accessor used across the framework — now resolves the Anthropic key from the vault (`sk-from-vault`), unwrapped only at the point of use with `.get_secret_value()`. The two tenant lookups then print their pinned models, proving that per-tenant construction coexists with the global injection without interference. Hosts that must *never* fall back to ambient environment can additionally set `config_source_strict`, turning a missing source into a `ConfigSourceError` instead of a silent `EnvConfigSource` default.

### Choosing between the two patterns

A useful rule of thumb: use `set_config_source()` when there is exactly one configuration for the whole process (a single-tenant server, a CLI run) — every one of the framework's `get_settings()` call sites then resolves through it transparently. Use `build_settings(MappingConfigSource(...))` when configurations must coexist — per tenant, per test, per request — and pass the resulting `Settings` object explicitly to whatever consumes it (the composition root's `build_runtime(settings, org_id=...)` being the canonical consumer). The two are complementary rather than exclusive, as `main()` above shows: a process can carry a global chain for its own identity while minting tenant-scoped `Settings` on demand.

## Key API

| Symbol | Role |
|---|---|
| `ConfigSourcePort` | `@runtime_checkable` Protocol — sync `load() -> Mapping[str, ConfigValue]`, must not raise; custom sources conform structurally |
| `ConfigValue` | `str \| SecretStr` — the value union a source may return |
| `ChainedConfigSource(sources)` | Ordered first-wins composite; a failing sub-source is logged and skipped |
| `MappingConfigSource(values)` | In-memory source from an explicit mapping (dict / tenant row / request payload) |
| `EnvConfigSource(dotenv_path=".env")` | Default adapter — env + `.env` + legacy `LIGHTAGENT_` mirror, placed last in the chain here |
| `set_config_source(source)` | Global injection (pattern 1); invalidates the `get_settings()` cache |
| `build_settings(source=None)` | Pure, per-tenant `Settings` constructor (pattern 2); `ContextVar`-isolated, no global mutation |
| `SecretStr` | Pydantic secret wrapper — masked in logs/`repr`, unwrapped via `.get_secret_value()` |

## Run it

```bash
uv run jupyter lab notebooks/config_source_custom.ipynb
uv run python examples/config_source_custom.py   # from the prismal repo
```

No API key is required — the "vault" is an in-memory dict, so everything runs offline. In the notebook, the final run cell ships as the commented `# main()`; uncomment it to execute. `main()` is a plain synchronous function, so no `await` is needed.

## Related

- [Injecting the default environment config source (Fase W)](config_source_env.md) — the baseline this notebook substitutes away from.
- [Runtime composition root (Fase R)](composition_root.md) — where per-tenant sources meet `build_runtime(org_id=...)` via `apply_org_overrides`.
- [Custom tool provider (Fase Y)](tool_provider_custom.md) — the same "plain class conforming a Protocol" pattern applied to tools.
