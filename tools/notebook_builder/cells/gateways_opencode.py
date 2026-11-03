from nbkit import header, md, code
CELLS = [
header("OpenCode Zen / Go gateway (Phase GW) — offline", "examples/gateways_opencode.py",
       "🔑 Not required — nothing leaves the process",
       """
       OpenCode exposes two gateways with different economics and rules:

       * **Zen** — pay-per-use, but each model family speaks its own **wire protocol** (Anthropic Messages, OpenAI Responses, Chat Completions).
       * **Go** — a flat monthly fee, usable **only by coding agents** (ToS), with rolling quota windows.

       The notebook shows:

       1. Per-model wire protocols resolved by the same router.
       2. The OpenCode Go ToS guard — non-coding agents are refused before any request.
       3. The Go flat-fee `QuotaLedger`: an exhausted 5 h window downgrades (or refuses) the call before any network call.
       """, "docs/gateways.md"),
md("## Setup · build the router from settings\n\n`build_gateway_router()` composes every enabled gateway for one tenant (`org_id`) and attaches the quota ledger."),
code('''
from __future__ import annotations

from pydantic import SecretStr

from prismal.core.config import Settings
from prismal.core.exceptions import GatewayPolicyError
from prismal.providers.gateways.router import build_gateway_router

settings = Settings(
    llm_gateways_enabled=True,
    opencode_zen_enabled=True,
    opencode_go_enabled=True,
    opencode_go_monthly_cap_usd=10.0,
    opencode_go_quota_action="downgrade",
    opencode_go_downgrade_model="gw:opencode-go/glm-5.3-flash",
)
router = build_gateway_router(
    settings, org_id="acme", secret_resolver=lambda ref: SecretStr("oc-demo-key")
)
assert router is not None
'''),
md("## 1 · Per-model wire protocols (Zen)"),
code('''
for ref in ("gw:opencode/claude-sonnet-4-5", "gw:opencode/gpt-5.5", "gw:opencode/glm-5.3"):
    resolved = router.resolve(ref, agent="researcher", session_id="sess-1")
    assert resolved is not None
    print(f"{ref:32} -> {resolved.litellm_model:34} [{resolved.wire_protocol}]")
'''),
md("## 2 · OpenCode Go ToS guard\n\nThe `coder` agent may use Go (`cost_mode=subscription` ⇒ metered `cost_usd=0` plus an `equivalent_usd`); `researcher` is refused."),
code('''
resolved = router.resolve("gw:opencode-go/glm-5.3", agent="coder", session_id="sess-1")
assert resolved is not None
print("coder      ->", resolved.litellm_model, "| cost_mode =", resolved.cost_mode)
print("headers    :", sorted(resolved.extra_headers))
try:
    router.resolve("gw:opencode-go/glm-5.3", agent="researcher", session_id="sess-1")
except GatewayPolicyError as exc:
    print("researcher -> refused:", exc.reason)
'''),
md("## 3 · Go quota ledger\n\nA $10/month cap gives a $2 budget per 5 h window. After recording $2 of equivalent usage, the preflight verdict for the next call is a downgrade to the configured cheaper model."),
code('''
ledger = router.ledger
assert ledger is not None
ledger.record("opencode-go", "glm-5.3", equivalent_usd=2.0)
verdict = ledger.preflight("opencode-go", "glm-5.3", est_equivalent_usd=0.01)
print(f"verdict: {verdict.action} (window {verdict.exhausted_window})")
print("downgrade target:", resolved.downgrade_model)
for snap in ledger.snapshot("opencode-go"):
    print(f"{snap.window:>5}: used ${snap.used:.2f} of ${snap.limit:.2f}")
'''),
md("---\n\nAfter a downgrade, `BudgetGuard.degradation()` reports `gateway_quota:<window>`, so the budget layer and the host read surface see the same event."),
]
