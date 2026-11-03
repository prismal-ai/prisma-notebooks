# The One-Way Human-Review Gate (DEC-HITL)

> **Notebook:** [`notebooks/decisions_hitl_gate.ipynb`](../notebooks/decisions_hitl_gate.ipynb) · **Source:** [`prismal/decisions/integrations/hitl.py`](https://github.com/prismal-ai/prismal/blob/main/prismal/decisions/integrations/hitl.py) · **Guide:** [`docs/decisions.md#human-review-gate-dec-hitl-oll-012014`](https://github.com/prismal-ai/prismal/blob/main/docs/decisions.md)

Before 3.19.0, `hitl_band()` asked a decision model one abstract question: *"may this action skip a human?"* Every Ollama model tested could be talked into "yes" by injection text in the action summary. The gate was redesigned on two principles:

- **Concrete questions.** Instead of one judgement call, the gate asks five single-fact Noul (yes/no) questions:
  - `hitl.destructive`
  - `hitl.rule_change`
  - `hitl.prior_approval`
  - `hitl.health_safety`
  - `hitl.money`
- **One-way.** A model's answer can *raise* the need for review but can never *clear* it on its own.

## How the band is derived

1. **Hard stops.** If the action is `irreversible`, its risk tier is `CRITICAL`, or the mode is `off` or `shadow`, the band is `human`.
2. **Rule tier.** `rule_flags()` checks the action without any model. Any of these flags gives `human`:
   - injection or authority-claim flags;
   - a tainted state;
   - destructive or money keywords in English or Spanish, including keywords inside `snake_case` and camelCase tool names;
   - a trusted `GatedAction.amount` above `decision_hitl_money_limit`.

   A model never compares amounts.
3. **Amount within the limit.** If the trusted amount is within the limit, the money fact is settled and `hitl.money` is not asked.
4. **Models.** The remaining questions go through the cascade:
   - Any answer at or above its threshold (`decision_hitl_raise_thresholds`) raises review, and the raise is sticky. Even raw answers from tiers that were later overruled count, via the `observe=` hook.
   - Any abstention gives `human`.
   - Five accepted, calibrated "no" answers from a backend other than `llm`, `ollama:fast` or `rule` give `auto`.
   - Confident but non-clearing "no" answers give `confirm`.

## What the notebook shows

| Scenario | Band |
|---|---|
| Calibrated backend, five "no" | `auto` |
| `ollama:fast`, five "no" | `confirm` |
| Uncalibrated backend, five "no" | `confirm` |
| `hitl.health_safety` P = 0.9 | `human` |
| One abstention | `human` |
| Refund $40, limit $100 (money not asked) | `auto` |
| Refund $900 | `human` (rule tier) |
| Irreversible / CRITICAL / shadow | `human` |

## Known limitation

If a calibrated tier is *persuaded* to answer "no" to all five questions, and the state carries no rule-tier signal, the gate still clears to `auto`. The defences are calibration with adversarial Spanish cases and the append-only injection corpus. In 3.22.0 the corpus gained 11 English and 30 Spanish phrasings, the authority-claim detector was widened to catch them, and it reports 0 false positives on the 655 calibration cases.
