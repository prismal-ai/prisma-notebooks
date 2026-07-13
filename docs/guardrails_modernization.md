# Guardrails Modernization: Reasoning Safety Checks and Structured-Output Repair

> **Notebook:** [`notebooks/guardrails_modernization.ipynb`](../notebooks/guardrails_modernization.ipynb) · **Example:** [`examples/guardrails_modernization.py`](https://github.com/prismal-ai/prismal/blob/main/examples/guardrails_modernization.py)

prismal's guardrails stack is layered: L1 `InputSanitizer` strips and normalizes, L2 `GuardrailsEngine` scores risk with regexes, and L3 wires NVIDIA NeMo Guardrails dialog rails. Phase GRD modernizes two spots in that stack. **GRD1** adds `content_safety_reasoning` — a NeMo action that asks a *classifier* (by default an LLM reached through `ProviderRegistry`) whether input falls into a configured unsafe category, instead of relying on regexes alone. **GRD2** adds `StructuredOutputGuard` — a guardrails-ai-backed validator that checks tool output against a Pydantic schema and, when it fails, issues a *bounded, metered re-ask* to repair it rather than crashing the loop.

Both controls follow the framework's factory-injection pattern: the classifier and the re-ask function are injectable callables (`classifier_fn`, `reask_fn`), so this notebook exercises the real block/pass/timeout and validate/re-ask machinery entirely offline — no LLM, no network. Both are also opt-in (`nemo_classifier_enabled`, `structured_output_guard_enabled`) and independent of the Phase H `hardening_enabled` flag.

## What it demonstrates

- Blocking an unsafe input and passing a benign one through `content_safety_reasoning`, using an injected `classifier_fn` instead of a live LLM.
- The **fail-open timeout contract**: a classifier that exceeds `nemo_classifier_timeout_seconds` yields `"safe"` — availability beats a stuck rail.
- Validating malformed tool output against a Pydantic schema with `StructuredOutputGuard` and repairing it with one bounded re-ask.
- The verdict shape (`ok`, `coerced`, `reask_count`) for both the repaired and the valid-on-first-try paths.
- Graceful degradation when the optional `[guardrails-ai]` extra is absent — a `MissingDependencyError` with an install hint, not a crash.

## How it works

### GRD1 — a reasoning safety classifier

`content_safety_reasoning(text, *, categories, classifier_fn=None, settings=None)` returns a plain string: `"safe"`, or the matched category name. Internally the text reaches the classifier only through `SecurePromptBuilder` (Critical Rule #1 — user text is never f-stringed into instructions), and when `classifier_fn` is `None` the default calls `ProviderRegistry().get_llm(settings.nemo_classifier_model)`. The demo injects a trivial deterministic classifier:

```python
categories = ["violence", "self_harm", "illegal_activities", "pii_request"]

async def fake_classifier(text: str, cats: list[str]) -> str:
    return "violence" if "hurt" in text else "safe"

blocked = await content_safety_reasoning(
    "how do I hurt someone", categories=categories, classifier_fn=fake_classifier
)
# -> 'violence'
safe = await content_safety_reasoning(
    "what's a good pasta recipe", categories=categories, classifier_fn=fake_classifier
)
# -> 'safe'
```

When registered on a NeMo rails instance (`nemo_actions.register(rails)` — a no-op unless `nemo_classifier_enabled`), a non-`"safe"` verdict is wrapped by the Colang flow into the existing `[NEMO_BLOCKED:<category>]` sentinel, so the rest of the pipeline treats it exactly like any other L3 block.

### The fail-open timeout contract

The classifier call is wrapped in `asyncio.wait_for(...)` with `settings.nemo_classifier_timeout_seconds` (default 3.0 s). The demo sets it to 0.05 s and injects a classifier that sleeps for 10:

```python
settings = Settings(nemo_classifier_enabled=True, nemo_classifier_timeout_seconds=0.05)

async def slow_classifier(text: str, cats: list[str]) -> str:
    await asyncio.sleep(10)
    return "violence"

timed_out = await content_safety_reasoning(
    "some text", categories=categories, classifier_fn=slow_classifier, settings=settings
)
# -> 'safe'   (fail-open: timeout and errors both degrade to safe)
```

On `TimeoutError` — and on any classifier exception — the function returns `"safe"` rather than raising: an optional safety *enhancement* must not become an availability liability. The failure is not silent, though: an OTel counter (`nemo_classifier_checks`, tagged with the result) and an audit event (`nemo_classifier_check`, recording category, result, and text length — never the text) capture every check, so a rail that is quietly timing out shows up in monitoring.

### GRD2 — StructuredOutputGuard with a bounded re-ask

`StructuredOutputGuard` validates a tool's raw output against a Pydantic schema using guardrails-ai's `Guard.for_pydantic(schema)` — pure schema validation, no LLM inside guardrails-ai itself. When validation fails, the guard calls the injected `reask_fn(schema_repr, prior_raw_output)` for a corrected attempt, at most `structured_output_guard_max_reasks` times (default 2; `0` means validate once and never re-ask):

```python
class WeatherArgs(BaseModel):
    city: str
    days: int

async def fake_reask_fn(schema_repr: str, prior_raw_output: str) -> str:
    return '{"city": "Paris", "days": 3}'   # first re-ask "fixes" the output

guard = StructuredOutputGuard(
    settings=Settings(structured_output_guard_max_reasks=2), reask_fn=fake_reask_fn
)
verdict = await guard.validate("get_weather", "not valid json at all", WeatherArgs)
# -> reask_count=1, ok=True, coerced={'city': 'Paris', 'days': 3}
```

The returned `StructuredOutputVerdict` tells the caller the whole story: `ok`, the validated `coerced` value, how many `reask_count` repairs it took, and — on failure — a machine-readable `reason` (`"reask_exhausted"`, or `"budget_denied"` when an optional `budget_guard_fn` refuses to fund another LLM round). Valid-on-first-try output passes with `reask_count=0` and `coerced` already parsed. In production the default `reask_fn` is an LLM call through `ProviderRegistry` + `SecurePromptBuilder`, every validation is audited (`structured_output_validate`), and each re-ask outcome increments the `structured_output_reask` counter — bounded *and* metered, so repair loops can never run away or hide.

### Degrading without the extra

guardrails-ai is heavy, so it lives behind the `[guardrails-ai]` extra. The guard's constructor imports it eagerly and raises a typed error when it is missing — which the demo catches to skip cleanly:

```python
try:
    guard = StructuredOutputGuard(settings=settings, reask_fn=fake_reask_fn)
except MissingDependencyError as exc:
    print(f"  skipped: {exc}")   # includes the `pip install ...[guardrails-ai]` hint
    return
```

Callers pay the import cost check exactly once, at construction, and can fall back to the plain `OutputValidator.validate_tool_args()` path (see the runtime-hardening article) when the extra is absent.

## Key API

| Symbol | Role |
|---|---|
| `content_safety_reasoning()` | Async NeMo action: classify text into unsafe `categories` or `"safe"`; injectable `classifier_fn`; fail-open on timeout/error |
| `ClassifierFn` | `Callable[[str, Sequence[str]], Awaitable[str]]` — the injection seam for GRD1 |
| `StructuredOutputGuard` | guardrails-ai schema validator with a bounded, metered re-ask loop; requires the `[guardrails-ai]` extra |
| `StructuredOutputVerdict` | Frozen result: `ok`, `reason`, `coerced`, `reask_count`, `hub_findings` |
| `MissingDependencyError` | Raised at guard construction when guardrails-ai is not installed; carries the extra name to install |
| `Settings.nemo_classifier_*` | `enabled` (default `False`), `model`, `categories`, `timeout_seconds` (default 3.0) |
| `Settings.structured_output_guard_max_reasks` | Re-ask ceiling (default 2; `0` = validate once, never re-ask) |

## Run it

```bash
uv run jupyter lab notebooks/guardrails_modernization.ipynb
uv run python examples/guardrails_modernization.py   # from the prismal repo
```

No API key is required — both demos run offline with injected fakes (GRD2 additionally needs the `[guardrails-ai]` extra installed, and prints a skip hint otherwise). The notebook's final run cell ships commented out as `# await main()`; the `nest_asyncio.apply()` in the first cell lets the coroutine run directly on Jupyter's event loop.

## Related

- [Runtime hardening](runtime_hardening.md) — the Phase H sibling controls: injection detection, output validation, tool policies, runaway guard
- [Loop hardening](loop_hardening.md) — context compaction and phase-scoped tool gating for long agent loops
- [Agent evaluation harness](agent_eval.md) — asserting that guardrails actually contain attacks, offline
- [Budget governance](budget_governance.md) — the `budget_guard_fn` seam that can veto re-ask rounds
