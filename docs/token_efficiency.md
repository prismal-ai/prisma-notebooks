# Token Efficiency: Measure First, Then Cut

> **Notebook:** [`notebooks/token_efficiency.ipynb`](../notebooks/token_efficiency.ipynb) · **Example:** [`examples/token_efficiency.py`](https://github.com/prismal-ai/prismal/blob/main/examples/token_efficiency.py) · **Guide:** [`docs/token-efficiency.md`](https://github.com/prismal-ai/prismal/blob/main/docs/token-efficiency.md)

Phase TKE (prismal 3.13.0) opened wave 2026-10. Its rule is *measure first, then cut*. The first part makes token accounting honest: cache reads, cache writes and reasoning tokens are tracked separately and priced as such. The second part adds controls that lower the token count without changing what agents do. Everything sits behind `token_efficiency_enabled` (default `False`). Every sub-flag works only when that master flag is on. With it off, the compiled graph, tool lists, `SecurePromptBuilder` output, `ChatLiteLLM` construction and metered `Usage` are identical to v3.12.0, and snapshot and contract tests check this.

## What it demonstrates

- **Cache-aware accounting (TKE1).** `CostMeter.record_response()` reads LangChain's `input_token_details`. With 18,000 of 20,000 input tokens served from cache, the cost drops from the legacy price to the cache-aware price, and the meter reports a 90 % hit ratio.
- **Cache-stable prompt layout (TKE2).** `PromptLayout.CACHE_STABLE` makes the system prompt byte-identical across requests:
  - an HMAC *prefix canary*, stable per tenant, stays in the system prompt;
  - the per-request canary moves to a tail `<turn_context>` message;
  - `detect_leak()` still catches either canary.
- **Prompt-cache breakpoints.** `apply_cache_policy()` detects the cache family (`anthropic_explicit` for Claude) and places `cache_control` breakpoints from `config/prompt_cache.yaml`. It never raises.
- **Tool-output budget (TKE5).** `ToolOutputBudget.apply()` caps a 15,000-token `web_search` result at its budget. The full 60,000-character payload is written to an `ArtifactStorePort` first, scoped by `(tenant, run_id)`.
- **TOON.** `maybe_toon()` encodes a uniform list of dicts as a compact header-plus-rows block.

## How it works

The LLM construction stack is fixed by ADR-WAVE-001, from inside out:

1. `GatewayChatLiteLLM`, innermost.
2. `CacheAwareChatModel`, which applies the cache policy and the per-role effort.
3. AMR, which only picks the model string.
4. `with_fallbacks`, outermost.

`compute_cost_usd` prices a call using the first source that has data: the gateway-billed cost, then LiteLLM's cache-aware price, then the `budget_pricing` table, then none.

On the tool side there are three more controls:

- **Cache-safe tool gating.** In `mask` mode the tool list sent to the model stays stable, so the prompt cache survives, and the phase allow-set is enforced when a tool is executed.
- **BM25 deferred tool loading.**
- **Observation masking.** In-loop masking only started running in 3.20.0. See [`observation_masking.md`](observation_masking.md).

## Measured result

TKE9-04 (3.20.0) measured the controls:

- With the controls tuned, tokens dropped 16.7 % on batched lookups and 21.4 % on sequential lookups, and the solve rate stayed at 1.0.
- With the default settings, nothing changes on these workloads.
- The token baseline is 2,723.5, and the `python -m prismal.eval token-gate` check now blocks regressions against it.
