from nbkit import header, md, code
CELLS = [
header("OpenRouter gateway (Phase GW) — offline, no request is sent", "examples/gateways_openrouter.py",
       "🔑 Not required — nothing leaves the process (the key is a fake `SecretStr`)",
       """
       Phase GW lets agents route LLM calls through gateways with `gw:<gateway>/<model>[:<variant>]` references.
       LiteLLM stays the only transport; everything is gated by `llm_gateways_enabled` plus one `*_enabled` flag per
       gateway. With the flags off, the graph and every `ProviderRegistry.get_llm()` call are identical to 3.13.0.

       For OpenRouter the notebook shows what prismal *would* send:

       1. The resolved LiteLLM target, headers and `extra_body` with the privacy defaults (`data_collection="deny"`, `zdr=true`) and a hashed sticky `session_id` (cache hits).
       2. Gateway-owned model fallback (`models[]`) switching off prismal's own fallback — **no double fallback**.
       3. A fail-closed policy gate refusing a variant outside the allow-list.
       4. Metering a gateway response, where the **billed** cost wins (`cost_source="gateway"`).
       """, "docs/gateways.md"),
md("## Setup · imports, settings and the gateway\n\nThe `secret_resolver` is where a vault lookup goes in production; keys are only unwrapped at the `GatewayChatLiteLLM` constructor and never reach `os.environ`, logs, URLs or `repr`."),
code('''
from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from pydantic import SecretStr

from prismal.budget.meter import CostMeter
from prismal.core.config import Settings
from prismal.core.exceptions import GatewayPolicyError
from prismal.providers.gateways.base import GatewayCallMeta
from prismal.providers.gateways.openrouter import OpenRouterGateway
from prismal.providers.gateways.router import GatewayRouter
from prismal.providers.registry import ProviderRegistry

settings = Settings(
    llm_gateways_enabled=True,
    openrouter_enabled=True,
    openrouter_app_url="https://example.org",
    openrouter_fallback_models=["deepseek/deepseek-chat"],
)
gateway = OpenRouterGateway(
    settings=settings,
    secret_resolver=lambda ref: SecretStr("sk-or-demo"),  # a vault lookup in production
    org_id="acme",
)
'''),
md("## 1 · Resolve a `gw:` reference\n\n`resolve()` is pure (no I/O). Note the privacy block in `extra_body` and that the API key never shows up in the object's `repr`."),
code('''
resolved = gateway.resolve(
    "gw:openrouter/anthropic/claude-sonnet-4.5", agent="researcher", session_id="sess-42"
)
print("litellm model:", resolved.litellm_model)
print("headers      :", dict(resolved.extra_headers))
print("extra_body   :", json.dumps(dict(resolved.extra_body), indent=2))
print("repr hides the key:", "sk-or-demo" not in repr(resolved))
'''),
md("## 2 · No double fallback\n\nWhen the gateway owns fallback (`openrouter_fallback_models`), the registry returns a single model with `max_retries=0` instead of wrapping it in prismal's own `with_fallbacks` chain — even though `fallback_model` is set."),
code('''
router = GatewayRouter({"openrouter": gateway}, settings=settings)
registry = ProviderRegistry(
    settings=settings.model_copy(update={"fallback_model": "gpt-4o-mini"}),
    gateway_router=router,
)
llm = registry.get_llm_with_fallback("gw:openrouter/anthropic/claude-sonnet-4.5")
print("returned:", type(llm).__name__, "| max_retries =", getattr(llm, "max_retries", None))
'''),
md("## 3 · Fail-closed policy gate\n\nPolicy, ToS and security guards fail **closed** before any request; here the `:free` variant is not in the allow-list."),
code('''
try:
    gateway.resolve("gw:openrouter/anthropic/claude-sonnet-4.5:free", agent=None, session_id=None)
except GatewayPolicyError as exc:
    print("refused:", exc.reason)
'''),
md("## 4 · Metering a gateway response\n\n`GatewayChatLiteLLM` stores the served model, generation id and billed cost under `response_metadata[\"prismal_gateway\"]`; `CostMeter` prefers that billed figure over any estimate."),
code('''
meta = GatewayCallMeta(
    gateway="openrouter",
    requested_model="anthropic/claude-sonnet-4.5",
    served_model="anthropic/claude-sonnet-4.5",
    generation_id="gen-demo",
    reported_cost_usd=0.0042,
    cost_mode="metered",
)
message = AIMessage(
    content="...",
    usage_metadata={"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500},
    response_metadata={"prismal_gateway": meta.to_dict()},
)
usage = CostMeter(settings=settings).record_response(message, resolved.litellm_model)
print(f"cost_usd={usage.cost_usd} source={usage.cost_source} gateway={usage.gateway}")
'''),
md("---\n\nNext: [`gateways_opencode.ipynb`](gateways_opencode.ipynb) (per-model wire protocols, ToS guard, flat-fee quota) and [`gateways_omniroute.ipynb`](gateways_omniroute.ipynb) (self-hosted gateway with a fail-closed startup gate)."),
]
