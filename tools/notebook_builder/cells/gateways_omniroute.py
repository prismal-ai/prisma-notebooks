from nbkit import header, md, code
CELLS = [
header("OmniRoute gateway (Phase GW) — fail-closed startup gate", "examples/gateways_omniroute.py",
       "🔑 Not required — the health probe is an `httpx.MockTransport`",
       """
       OmniRoute is **self-hosted**, so prismal will not trust it blindly. Before any model can be resolved the router
       runs a startup gate that needs an API key, an upstream attestation (`omniroute_upstream_attestation`) and a
       version floor for CVE-2026-88062. Until `astartup()` succeeds — or if the gate itself crashes — OmniRoute stays
       disabled (fail closed).

       1. Version 3.8.50 (vulnerable) is refused and the router disables the gateway.
       2. Version 3.8.51 (fixed) passes and the gateway resolves models.
       """, "docs/gateways.md"),
md("## Setup · a gateway factory with a fake `/api/monitoring/health`"),
code('''
from __future__ import annotations

import httpx
from pydantic import SecretStr

from prismal.core.config import Settings
from prismal.core.exceptions import GatewayPolicyError
from prismal.providers.gateways.omniroute import OmniRouteGateway
from prismal.providers.gateways.router import GatewayRouter


def make_gateway(settings: Settings, version: str) -> OmniRouteGateway:
    def health(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"version": version})

    return OmniRouteGateway(
        settings=settings,
        secret_resolver=lambda ref: SecretStr("omni-demo-key"),
        http=httpx.AsyncClient(transport=httpx.MockTransport(health)),
    )


settings = Settings(
    llm_gateways_enabled=True,
    omniroute_enabled=True,
    omniroute_upstream_attestation="api_keys_only",
)
'''),
md("## 1 · OmniRoute 3.8.50 (affected by CVE-2026-88062)"),
code('''
router = GatewayRouter({"omniroute": make_gateway(settings, "3.8.50")}, settings=settings)
await router.astartup()
print("disabled:", router.disabled)
try:
    router.resolve("gw:omniroute/my-combo", agent=None, session_id=None)
except GatewayPolicyError as exc:
    print("resolve refused:", exc.reason)
await router.aclose()
'''),
md("## 2 · OmniRoute 3.8.51 (fixed)"),
code('''
router = GatewayRouter({"omniroute": make_gateway(settings, "3.8.51")}, settings=settings)
await router.astartup()
resolved = router.resolve("gw:omniroute/my-combo", agent=None, session_id=None)
assert resolved is not None
print("litellm model:", resolved.litellm_model, "| api_base:", resolved.api_base)
await router.aclose()
'''),
md("---\n\nIn a host, `build_runtime(..., gateway_vault=...)` composes `RuntimeContext.gateways` per tenant and runs `astartup()` for you."),
]
