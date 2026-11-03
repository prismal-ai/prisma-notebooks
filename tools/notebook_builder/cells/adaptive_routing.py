from nbkit import header, md, code
CELLS = [
header("Adaptive model routing with fakes (Phase AMR)", "examples/adaptive_routing.py",
       "🔑 Not required — no API key, no LLM call (only model *strings* are chosen)",
       """
       Phase AMR adds a `RoutingPolicyPort` that picks the **model string** for each agent call. It never wraps the
       model (ADR-WAVE-001, step 1): prismal picks *which model*, gateways pick *which provider*. Everything is gated by
       `adaptive_routing_enabled` (default `False`) and starts in `routing_mode="shadow"` (the choice is recorded but
       the old model is used).

       1. **Parity** — the shipped tier map keeps every agent on `default_model`.
       2. **Cascade + affinity** — `coder` starts small, escalates after tool errors, and stays there (cache affinity).
       3. **Security floors** — a critical-risk turn is clamped to `frontier`, whatever the policy says.
       4. **Budget** — over the soft cap, the budget-aware policy drops one tier.
       5. **Batch assignment** — Skynet-style sub-orders assigned jointly under an accuracy floor (OmniRouter-style Lagrangian dual).

       Invariants: policies fail **open** (static fallback); security floors fail **closed**; an explicit `model=`
       always beats routing.
       """, "docs/routing.md"),
md("## Setup · a tier map and a success predictor\n\nThe tier map binds tiers to models, pins per-agent tiers and declares risk floors. The predictor holds observed success rates per (intent, tier) for the batch assigner."),
code('''
from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from prismal.budget.types import Budget, BudgetStatus, Usage
from prismal.core.config import Settings
from prismal.decisions.types import RiskTier
from prismal.routing.fanout import assign_fanout, build_batch_policy, fanout_features
from prismal.routing.policies.budget_aware import BudgetAwarePolicy
from prismal.routing.policies.static import StaticTierPolicy
from prismal.routing.select import clear_routing_run, record_outcome, select_model
from prismal.routing.types import RoutingOutcome, TaskFeatures

TIERS = """\\
tiers: {small: anthropic/claude-haiku-4-5, medium: anthropic/claude-sonnet-4-5,
        frontier: anthropic/claude-opus-4-5}
agents: {coder: frontier, researcher: medium}
risk_floor: {high: medium, critical: frontier}
"""

PREDICTOR = """\\
{"version": "demo", "cells": [
  {"intent": "_", "tier": "small", "successes": 9, "trials": 10},
  {"intent": "_", "tier": "frontier", "successes": 10, "trials": 10}]}
"""

work = Path(tempfile.mkdtemp(prefix="amr-demo-"))
tiers = work / "tiers.yaml"
tiers.write_text(TIERS, encoding="utf-8")


def make_state(sid: str, text: str = "fix the failing test", **routing: object) -> dict[str, object]:
    return {
        "session_id": sid,
        "messages": [HumanMessage(content=text)],
        "metadata": {"routing": dict(routing)} if routing else {},
    }
'''),
md("## 1 · Parity map (shipped `config/routing_tiers.yaml`)\n\nTurning the layer on with the shipped map changes nothing until you fill in tiers."),
code('''
parity = Settings(default_model="claude-sonnet-4-5")
static = StaticTierPolicy(settings=parity)
for agent in ("coder", "file_manager", "researcher"):
    features = TaskFeatures(agent, None, 100, True, False)
    print(f"{agent:13s} -> {static.select(features).model_ref}")
'''),
md("## 2 · Cheapest-first cascade + cache affinity\n\n`select_model` records the choice under `metadata[\"routing\"]` in a per-run registry keyed by `session_id`. `record_outcome` reports two tool errors ⇒ escalation; the next turn keeps the escalated tier so the provider prompt cache stays warm."),
code('''
cascade = Settings(
    adaptive_routing_enabled=True,
    routing_mode="enforce",
    routing_policy="cascade",
    routing_tier_models=str(tiers),
)
state = make_state("demo-cascade")
first = select_model(state, "coder", settings=cascade)
escalated = record_outcome(state, "coder", RoutingOutcome(success=True, tool_errors=2))
state["messages"].append(HumanMessage(content="next request"))  # type: ignore[union-attr]
later = select_model(state, "coder", settings=cascade)
print(f"turn 1: {first.tier} ({first.model_ref})")
print(f"escalation: {escalated.reason if escalated else None}")
print(f"turn 2 (affinity): {later.tier} ({later.model_ref})")
clear_routing_run(state)
'''),
md("## 3 · Security floor\n\nA `critical` risk tier forces `frontier` — floors are fail-closed, and a deployment without a `critical` floor refuses to start."),
code('''
critical = make_state("demo-floor", risk_tier="critical")
choice = select_model(critical, "researcher", settings=cascade)
print(f"critical researcher -> {choice.tier} ({choice.reason})")
clear_routing_run(critical)
'''),
md("## 4 · Budget soft cap\n\n`BudgetAwarePolicy` wraps any policy and downgrades one tier when the `BudgetStatus` reports the soft cap was crossed."),
code('''
budget = BudgetAwarePolicy(StaticTierPolicy(settings=cascade), settings=cascade)
status = BudgetStatus(False, True, False, "tokens", Usage(), Budget(max_tokens=1000))
features = TaskFeatures("coder", None, 100, True, False, risk_tier=RiskTier.LOW)
print(f"coder over soft cap -> {budget.select(features, budget=status).tier}")
'''),
md("## 5 · Batch assignment for a fan-out round\n\nFour sub-orders are assigned **jointly**: the cheap tier's predicted success (0.9) already meets the accuracy floor, so all go to `small`."),
code('''
predictor = work / "predictor.json"
predictor.write_text(PREDICTOR, encoding="utf-8")
batch = cascade.model_copy(
    update={
        "routing_policy": "batch",
        "routing_predictor_path": str(predictor),
        "budget_pricing": {
            "anthropic/claude-haiku-4-5": {"input": 0.001, "output": 0.005},
            "anthropic/claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
            "anthropic/claude-opus-4-5": {"input": 0.005, "output": 0.025},
        },
    }
)
items = [(f"ord-{i}", fanout_features("skynet_worker", f"sub-task {i}")) for i in range(4)]
choices = assign_fanout(items, build_batch_policy(batch), batch) or {}
for item_id, c in choices.items():
    print(f"{item_id} -> {c.tier} ({c.model_ref})")
'''),
md("---\n\nAlso in `prismal/routing/`: an intent-keyed plan cache, an exact response cache and a provider `BatchExecutor` (Anthropic / OpenAI / OpenRouter `:batch`). Tainted or HITL-gated requests are never batched. The cascade's adequacy check is a typed DEC decision — see [`typed_decisions.ipynb`](typed_decisions.ipynb)."),
]
