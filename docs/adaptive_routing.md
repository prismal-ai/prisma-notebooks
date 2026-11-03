# Adaptive Model Routing

> **Notebook:** [`notebooks/adaptive_routing.ipynb`](../notebooks/adaptive_routing.ipynb) · **Example:** [`examples/adaptive_routing.py`](https://github.com/prismal-ai/prismal/blob/main/examples/adaptive_routing.py) · **Guide:** [`docs/routing.md`](https://github.com/prismal-ai/prismal/blob/main/docs/routing.md)

Phase AMR (3.16.0) adds a `RoutingPolicyPort` that picks the **model string** for each agent call. It never wraps the model; that is step 1 of the ADR-WAVE-001 pipeline. The split of responsibilities is: prismal picks *which model*, gateways pick *which provider*. The layer is gated by `adaptive_routing_enabled` (default `False`) and starts in `routing_mode="shadow"`. In shadow mode the choice is recorded under `metadata["routing"]`, but the 3.15.0 model is still used.

## What it demonstrates

1. **Parity.** The shipped `config/routing_tiers.yaml` keeps every agent on `default_model` until you fill in tiers.
2. **Cascade and cache affinity.** `coder` starts on the small tier. `record_outcome(..., tool_errors=2)` escalates it to medium (`escalation:tool_errors`), and affinity keeps it on medium for the next turn so the provider's prompt cache stays warm. Other escalation triggers are structured-output failures and inadequate answers; adequacy is judged by a typed DEC Noul check.
3. **Security floor.** A `critical` risk tier is clamped to `frontier` (`risk_floor`). Floors fail **closed**, and a deployment without a `critical` floor refuses to start.
4. **Budget.** `BudgetAwarePolicy` wraps any policy and drops one tier once the `BudgetStatus` reports that the soft cap was crossed.
5. **Batch assignment.** For Skynet or MoA fan-outs, `assign_fanout()` assigns all sub-orders jointly. It uses a Lagrangian dual under an accuracy floor, with a greedy fallback, and never raises. The predictor estimates 0.9 success for the small tier, which meets the floor, so all four orders go there.

## Invariants

- Policies fail **open** (static `fallback:<err>`). Security floors fail **closed**.
- Tainted state is never routed below the static tier.
- An explicit `model=` always beats routing.
- `routing/` imports no provider SDK; provider batch APIs live only in `providers/batch_adapters.py`.
- Tainted or HITL-gated requests are never batched.

Other parts of `prismal/routing/`: an intent-keyed plan cache, an exact response cache and a `BatchExecutor` for Anthropic, OpenAI and OpenRouter `:batch`.
