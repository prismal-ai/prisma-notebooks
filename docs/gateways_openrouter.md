# LLM Gateways I: OpenRouter

> **Notebook:** [`notebooks/gateways_openrouter.ipynb`](../notebooks/gateways_openrouter.ipynb) · **Example:** [`examples/gateways_openrouter.py`](https://github.com/prismal-ai/prismal/blob/main/examples/gateways_openrouter.py) · **Guide:** [`docs/gateways.md`](https://github.com/prismal-ai/prismal/blob/main/docs/gateways.md)

Phase GW (3.14.0) lets any agent route through a gateway by naming a model as `gw:<gateway>/<model>[:<variant>]`. A model is chosen in this order: an explicit model, then `llm_gateway_agent_map[agent]`, then the legacy setting. LiteLLM stays the only transport. The layer is gated twice: `llm_gateways_enabled` turns it on, and each gateway has its own `*_enabled` flag. With both off, the kwargs of every `ProviderRegistry.get_llm()` call are identical to 3.13.0. The shared contract test `tests/unit/providers/test_registry_contract.py` checks this.

## What it demonstrates

1. **`resolve()` is pure.** It does no I/O and returns a `ResolvedModel`. The OpenRouter adapter always sends privacy defaults: `provider.data_collection="deny"` and `zdr=true`. It also sends a hashed sticky `session_id`, so repeated calls hit the provider's prompt cache. The key is a `SecretStr` with `repr=False`.
2. **No double fallback.** If the gateway owns fallback (`models[]`), `get_llm_with_fallback()` returns one `GatewayChatLiteLLM` with `max_retries=0` and does not wrap it in prismal's own fallback chain.
3. **Fail-closed policy.** A variant outside the allow-list, `:free` here, is refused with `GatewayPolicyError(reason="variant_not_allowed")` before any request.
4. **Billed cost wins.** `GatewayChatLiteLLM` stores the served model, generation id and billed cost under `response_metadata["prismal_gateway"]`. `CostMeter` uses that billed cost and reports `cost_source="gateway"`. `CostReconciler` can later reconcile it against the gateway's generation API.

## Invariants worth remembering

- Keys are unwrapped only inside the `GatewayChatLiteLLM` constructor. They never reach `os.environ`, logs, URLs or `repr`.
- Policy, ToS and security guards fail **closed**. Lookups such as pricing and model lists fail **open**.
- In a host, `build_runtime(..., gateway_vault=...)` gives each tenant its own keys (`<auth_ref>@<org_id>`). There is no global fallback key unless `llm_gateway_tenant_key_fallback` is set.
