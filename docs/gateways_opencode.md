# LLM Gateways II: OpenCode Zen / Go

> **Notebook:** [`notebooks/gateways_opencode.ipynb`](../notebooks/gateways_opencode.ipynb) · **Example:** [`examples/gateways_opencode.py`](https://github.com/prismal-ai/prismal/blob/main/examples/gateways_opencode.py) · **Guide:** [`docs/gateways.md`](https://github.com/prismal-ai/prismal/blob/main/docs/gateways.md)

OpenCode runs two gateways with different rules. The model catalogue for both is `config/opencode_models.yaml`, regenerated from the 2026-10 docs in 3.20.0.

- **Zen** bills per use. Each model family speaks its own wire protocol.
- **Go** is a flat-fee subscription. Its ToS allow only coding agents, and usage is limited by rolling quota windows.

## What it demonstrates

1. **Per-model wire protocols.** One `GatewayRouter` maps Claude models to the Anthropic Messages API (`anthropic_messages`), GPT models to the Responses API (`openai_responses`) and GLM models to Chat Completions (`openai_chat`).
2. **ToS guard.** `coder` may use `gw:opencode-go/...`, which resolves with `cost_mode="subscription"`. `researcher` is refused (`opencode_go_agent_not_allowed`) before any request is sent.
3. **Quota ledger.** `QuotaLedger` tracks Go's flat fee as *equivalent USD* per window. With `opencode_go_monthly_cap_usd=10`, the windows are $2 per 5 h, $5 per week and $10 per month. Once $2 has been recorded, `preflight()` returns `downgrade` for the 5 h window, and the call goes to `opencode_go_downgrade_model`. Setting `opencode_go_quota_action="refuse"` refuses the call instead.

## Notes

- Subscription calls are metered with `cost_usd=0` plus an `equivalent_usd` figure. After a downgrade, `BudgetGuard.degradation()` reports `gateway_quota:<window>`.
- `opencode_denied_models` refuses free tiers whose data may be used to train the upstream model (`big-pickle`, `exo-free`, `mimo-*-free`, …). It also refuses NVIDIA's "trial use only" endpoints. The list was re-verified on 2026-10-12.
