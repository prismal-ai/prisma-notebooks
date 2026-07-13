# Node I/O type-safety — off, warn, enforce (Phase NTS)

> **Notebook:** [`notebooks/node_typesafety.ipynb`](../notebooks/node_typesafety.ipynb) · **Example:** [`examples/node_typesafety.py`](https://github.com/prismal-ai/prismal/blob/main/examples/node_typesafety.py)

Phase NTS adds an additive, opt-in **I/O contract layer** to `@prismal_node`: a node may declare a narrow `input_model` and/or `output_model` (plain Pydantic `BaseModel`s) describing the *subset* of `AgentState` it reads and the subset of the state update it writes. The design is deliberately non-invasive — both parameters default to `None`, the validation helpers are pure and never raise on their own, and the whole feature sits behind `settings.node_typesafety_enabled` (default `False`), so un-annotated nodes and disabled installs are byte-for-byte unchanged.

The payoff is catching a class of bug LangGraph cannot: a node that silently forgets a key in its return dict (say, `current_agent`) breaks routing downstream with no error at the source. With an output contract declared, that omission surfaces exactly where it happens — logged and counted in `warn` mode, or turned into a structured `metadata.error` update in `enforce` mode.

## What it demonstrates

- Declaring `input_model` / `output_model` on `@prismal_node` as narrow Pydantic projections of `AgentState` — only the declared keys are validated; everything else in state is ignored.
- The three control modes of `settings.node_typesafety_mode` on the same deliberately buggy node: `off`/disabled (silent passthrough), `warn` (log + OTel counter, data passes through unmodified), `enforce` (validation failure raises `NodeValidationError`).
- How `enforce`-mode failures surface to graph code: the outermost `error_mapping_middleware` catches the exception and returns a `{"metadata": {"error": {...}}}` state update instead of crashing the graph.
- That a conforming node passes cleanly even under `enforce` — the contract costs nothing when it is met.

## How it works

### Declaring the contracts

The contracts are ordinary Pydantic models whose field names are a literal 1:1 subset of `AgentState`'s keys — a *narrow projection*, not an exhaustive mirror. Validation reads only the declared keys; keys the state has but the model does not declare are never inspected:

```python
class GreeterInput(BaseModel):
    """The toy node reads a session id and a message list."""

    session_id: str
    messages: list


class GreeterOutput(BaseModel):
    """The toy node must return which agent ran and the new messages."""

    current_agent: str
    messages: list
```

One security convention is worth knowing: when validation fails, the error messages carry field *names* and Pydantic's own type/constraint description only — never the offending field *value* — so sensitive state content cannot leak into logs, metrics, or the audit trail.

### A buggy node under contract

The demo decorates a node that deliberately forgets `current_agent` in its return — in a real supervisor graph this is a silent routing break. The output contract makes it observable:

```python
@prismal_node(
    name="greeter",
    security="off",
    audit=False,
    input_model=GreeterInput,
    output_model=GreeterOutput,
)
async def greeter_node(state):
    # Deliberately BUGGY: forgets ``current_agent`` — a silent routing break
    # in production, caught by the output contract here.
    return {"messages": [f"hello {state['session_id']}"]}
```

Inside the decorator's middleware chain, `node_io_validation_middleware` is the **innermost** entry — it wraps the user function directly, so input validation sees exactly the state the node receives and output validation sees exactly what it returned, before any other middleware touches either.

### Switching modes through settings

The demo swaps `Settings` via a module-level test seam (`_middleware.get_settings`), which is just a compact stand-in for however a host configures settings:

```python
def _use_settings(**kwargs: object) -> None:
    """Point the middleware at a fresh Settings (module-level test seam)."""
    settings = Settings(**kwargs)
    _middleware.get_settings = lambda: settings
```

Two knobs govern behaviour. `node_typesafety_enabled` (default `False`) is the master opt-in: when it is off, the middleware is a pure passthrough and validation code never runs. `node_typesafety_mode` (`off | warn | enforce`, default `warn`) then decides what a failure means; an unknown mode is rejected at settings load time by a model validator, so a typo fails fast rather than silently disabling the control.

### The three failure behaviours

The demo runs the same buggy node under each configuration:

```python
print("── disabled (default) — buggy node passes silently ──")
_use_settings(node_typesafety_enabled=False)
print("  ", await greeter_node(state))

print("── warn — buggy output logged + counted, still passes through ──")
_use_settings(node_typesafety_enabled=True, node_typesafety_mode="warn")
print("  ", await greeter_node(state))

print("── enforce — buggy output mapped to metadata.error ──")
_use_settings(node_typesafety_enabled=True, node_typesafety_mode="enforce")
out = await greeter_node(state)
print("  ", out.get("metadata", {}).get("error", out))
```

- **Disabled / `off`** — the decorated node behaves exactly like an undecorated one; the malformed return passes through untouched.
- **`warn`** — validation runs; the failure emits a `node_io_validation_failed` structured log and increments the `node_io_validation_failures` OTel counter (every validation also bumps `node_io_validated`), then the original, non-conforming update passes through unmodified. This is the recommended rollout mode: full visibility, zero behaviour change.
- **`enforce`** — validation raises `NodeValidationError` carrying the node name, the update's key names, the direction (`input` or `output`), and the field-level `schema_errors`. Because `NodeValidationError` *is-a* `NodeExecutionError`, the outermost `error_mapping_middleware` catches it unchanged and — with the decorator's default `raise_on_error=False` — converts it into a `{"metadata": {"error": {...}}}` state update, so the graph keeps running and the supervisor can react to a structured error instead of an exception.

The final section runs a correct node (`greeter_ok_node`, which does return `current_agent`) under `enforce` and shows it passing cleanly — the contract only bites when broken.

## Key API

| Symbol | Role |
|---|---|
| `@prismal_node(..., input_model=, output_model=)` | Declare optional Pydantic contracts for the state subset a node reads/writes; both default to `None` (no validation). |
| `settings.node_typesafety_enabled` | Master opt-in (default `False`); off ⇒ the validation middleware is a pure passthrough. |
| `settings.node_typesafety_mode` | `off \| warn \| enforce` (default `warn`); invalid values rejected at settings load. |
| `node_io_validation_middleware` | Innermost middleware; validates input before and output after the user function. |
| `validate_node_input` / `validate_node_output` | Pure, never-raising helpers returning a `NodeIOValidationResult` (`ok`, `direction`, field-name-only `errors`). |
| `NodeValidationError` | Raised in `enforce` mode; subclasses `NodeExecutionError`, so `error_mapping_middleware` maps it to a `metadata.error` update. |

## Run it

```bash
uv run jupyter lab notebooks/node_typesafety.ipynb
uv run python examples/node_typesafety.py   # from the prismal repo
```

No API key is required — the nodes are toy async functions and everything runs offline. In the notebook, the first cell applies `nest_asyncio`, and the final cell contains a commented `# await main()`; uncomment it and run the cell to walk all four scenarios (disabled, warn, enforce-buggy, enforce-clean) in sequence.

## Related

- [Runtime hardening](runtime_hardening.md) — the neighbouring `hardening_middleware` in the same `@prismal_node` chain.
- [Loop hardening](loop_hardening.md) — bounding iteration where type-safety bounds shape.
- [Guardrails modernization](guardrails_modernization.md) — content-level validation, complementary to structural I/O contracts.
- [Observability integration](observability_integration.md) — where the `node_io_*` counters and structured logs land.
