from nbkit import header, md, code
CELLS = [
header("Token efficiency with injected fakes (Phase TKE)", "examples/token_efficiency.py",
       "🔑 Not required — runs offline (no LLM, no network)",
       """
       Phase TKE is the first layer of wave 2026-10: **measure first, then cut tokens** without changing
       agent behaviour. Everything is gated by `token_efficiency_enabled` (default `False`); every sub-flag only
       takes effect with the master flag on, and with it off the graph, tool lists, prompts and metered `Usage`
       are identical to v3.12.0.

       This notebook walks the five building blocks of the original example, one cell each:

       1. **Cache-aware accounting** — the same cache-heavy response priced the v3.12 way and the cache-aware way.
       2. **Cache-stable prompt layout** — a byte-identical system prompt across requests, with both canaries still detectable.
       3. **Prompt-cache breakpoints** — where the Anthropic policy places `cache_control`.
       4. **Tool-output budget + artifact store** — a huge tool output is capped; the full payload stays retrievable.
       5. **TOON** — a compact encoding for uniform tables.
       """, "docs/token-efficiency.md"),
md("## Setup · imports and two `Settings` profiles\n\n`off` is the default profile (TKE disabled); `on` turns on the master flag plus the sub-flags this demo needs."),
code('''
from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from prismal.agents.artifacts import InMemoryArtifactStore
from prismal.agents.tool_output_budget import ToolOutputBudget
from prismal.budget.meter import CostMeter
from prismal.core.config import Settings
from prismal.providers.prompt_cache import CachePolicy, apply_cache_policy
from prismal.security.prompt_builder import SecurePromptBuilder
from prismal.utils.toon import maybe_toon

MODEL = "claude-sonnet-4-5"

off = Settings()
on = Settings(
    token_efficiency_enabled=True,
    prompt_layout="cache_stable",
    prompt_canary_secret="demo-secret",  # noqa: S106 — offline demo value
    tool_output_budget_enabled=True,
    tool_output_max_tokens_default=200,
    tool_output_toon_enabled=True,
)
'''),
md("""
## 1 · Cache-aware accounting (TKE1)

A fake response reports 20,000 input tokens, 18,000 of which were **cache reads**. The legacy meter prices every
input token at the full rate; with TKE on, `CostMeter` reads LangChain's `input_token_details` and prices cache reads at
the provider's discounted rate (gateway → LiteLLM cache-aware → pricing table → none, per ADR-WAVE-001).
"""),
code('''
response = SimpleNamespace(
    usage_metadata={
        "input_tokens": 20_000,
        "output_tokens": 500,
        "input_token_details": {"cache_read": 18_000},
    },
    response_metadata={},
)
legacy = CostMeter(settings=off).record_response(response, MODEL)
aware = CostMeter(settings=on).record_response(response, MODEL, agent="coder")
print(
    f"[1] v3.12 price ${legacy.cost_usd:.4f} → cache-aware ${aware.cost_usd:.4f} "
    f"(hit ratio {aware.cache_hit_ratio:.0%})"
)
'''),
md("""
## 2 · Cache-stable prompt layout (TKE2)

Prompt caching only pays off if the **prefix is byte-identical** between requests. With
`prompt_layout="cache_stable"`, `SecurePromptBuilder` puts an HMAC-derived *prefix canary* (stable per tenant) in the
system prompt and moves the per-request canary into a tail `<turn_context>` message. Leak detection still covers both
canaries — the red-team leak parity is a hard invariant of this layout.
"""),
code('''
first, second = SecurePromptBuilder(settings=on), SecurePromptBuilder(settings=on)
m1 = first.build("You are a careful assistant.", "hello", tenant="acme")
m2 = second.build("You are a careful assistant.", "bye", tenant="acme")
leak = first.detect_leak(f"oops {first.prefix_canary('acme')}")
print(
    f"[2] system prompt identical: {m1[0]['content'] == m2[0]['content']}; "
    f"prefix-canary leak detected: {leak.leaked}"
)
'''),
md("## 3 · Prompt-cache breakpoints (TKE2)\n\n`apply_cache_policy` detects the cache family from the model string and places breakpoints according to `config/prompt_cache.yaml`. It never raises: an unknown family simply gets no breakpoints."),
code('''
applied = apply_cache_policy(
    [SystemMessage(content="Static rules. " * 600), HumanMessage(content="hi")],
    [],
    model=MODEL,
    agent="coder",
    policy=CachePolicy(),
)
print(f"[3] family={applied.family} breakpoints={applied.breakpoints}")
'''),
md("""
## 4 · Tool-output budget + artifact store (TKE4/TKE5)

A `web_search` result of ~60,000 characters is capped to `tool_output_max_tokens_default` (200 tokens). The full payload
goes to an `ArtifactStorePort` scoped by `(tenant, run_id)` first, so nothing is lost — the agent can page it back
(see [`observation_masking.ipynb`](observation_masking.ipynb) for the in-loop `read_artifact` tool).
"""),
code('''
store = InMemoryArtifactStore(settings=on)
budgeted = ToolOutputBudget(settings=on, artifact_store=store).apply(
    "web_search", "lorem ipsum " * 5_000, run_id="demo", tenant="acme"
)
assert budgeted.artifact is not None
full = store.get(budgeted.artifact.id, run_id="demo", tenant="acme")
print(
    f"[4] {budgeted.original_tokens} tokens → {len(budgeted.content)} chars; "
    f"artifact holds {full.total_chars} chars"
)
'''),
md("## 5 · TOON encoding\n\nUniform lists of dicts are re-encoded as TOON (a header row + values) when that is shorter than JSON."),
code('''
rows = [{"id": 1, "city": "Caracas"}, {"id": 2, "city": "Madrid"}]
print("[5]", maybe_toon(rows).replace("\\n", " | "))
'''),
md("""
---

## Where this plugs into the graph

| Control | Setting(s) | Seam |
|---|---|---|
| Cache-aware `Usage` | `token_efficiency_enabled` | `CostMeter.record_response` |
| Cache-stable prompt | `prompt_layout="cache_stable"`, `prompt_canary_secret` | `SecurePromptBuilder.build` |
| Prompt caching | `config/prompt_cache.yaml` | `CacheAwareChatModel` (outer LLM wrapper) |
| Tool-output budget | `tool_output_budget_enabled`, `tool_output_max_tokens_default` | `react_loop` + `tool_output_budget_middleware` |
| Observation masking | `context_compaction_strategy="mask_observations"` | `ContextCompactor` inside every agent's `react_loop` (3.20.0) |

Measured result (TKE9-04, 3.20.0): tuned controls cut −16.7 % (batched lookups) and −21.4 % (sequential lookups) with
solve rate 1.0; the defaults change nothing on those workloads.
"""),
]
