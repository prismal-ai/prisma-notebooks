# Typed Decision Models: Closed Questions Without the LLM

> **Notebook:** [`notebooks/typed_decisions.ipynb`](../notebooks/typed_decisions.ipynb) · **Example:** [`examples/typed_decisions.py`](https://github.com/prismal-ai/prismal/blob/main/examples/typed_decisions.py) · **Guide:** [`docs/decisions.md`](https://github.com/prismal-ai/prismal/blob/main/docs/decisions.md)

Many agent steps ask a closed question: which route to take, whether a chunk is relevant, what score to give, whether a person needs to review. Phase DEC (3.15.0, extended in 3.19.0–3.22.0) answers these questions with *typed* decision models. A question is one of three kinds:

- **Choice**: pick one of the listed options.
- **Score**: give a value on a scale.
- **Noul**: answer yes or no.

The answer comes from a cascade behind `DecisionPort`, tried in this order: rule hook → `ollama:fast` → `ollama:accurate` → Jev → LLM. Laya is an optional extra tier. Only if every typed tier declines does the general LLM path run. The design rule is **a decision proposes; deterministic code authorizes.**

## What it demonstrates

1. **Calibration.** `build_calibration_record()` fits temperature scaling for an over-confident backend: it says p = 0.99 but is right about 97 % of the time. The fit lowers ECE and derives the acceptance threshold `tau` for a target precision of 0.95. `CalibrationStore` pins each record to the model it was fitted for. In `enforce` mode, `Settings` refuses to load unless a valid record exists.
2. **Accepted route.** A calibrated Laya stand-in answers `route.supervisor`, and `decide_route()` returns `coder`. In the graph this means the supervisor makes no routing LLM call.
3. **Taint ⇒ escalation.** `DecisionStateBuilder` is the only way untrusted text reaches a decision model. It flags "NOTE TO REVIEWER: … pre-approved" as an `authority_claim` and taints the state. `decide_route()` then returns `None`, which hands the turn back to the LLM supervisor.
4. **Option-order shuffle.** For HIGH and CRITICAL Choice questions the gate asks again with the options reordered so no option keeps its position (a derangement). A backend biased toward the first option answers inconsistently and is escalated with reason `shuffle_inconsistent`.

## Rollout and state of the art (3.22.x)

- **Rollout.** Turning a seam on takes `decisions_enabled` plus one `decision_<integration>_enabled` flag. Each integration starts in `shadow` mode. `promotion_ready` requires 2,000 decisions or 14 days of shadow data, and drift automatically demotes an integration.
- **Local default backend (3.19.0).** Ollama serves the cascade as two sub-tiers, `tev1:4b` (fast) and `nimble:9b-q4_K_M` (accurate). Each has its own time budget. In 3.22.0, `decision_ollama_timeout_ms_by_tier` was added so the accurate tier can have a longer budget than the fast one.
- **Calibration (DEC3-07, 3.22.0).** `route.supervisor` and `rag.chunk_relevance` are calibrated on both sub-tiers. Each was fitted on 300+ blind-reviewed Spanish cases, and both meet the `enforce` bar (ECE ≤ 0.05).
- **Audit.** Audit records are hash-only (`decision.made`).
- **Injection corpus.** The EN/ES corpus is append-only.

See also [`decisions_hitl_gate.md`](decisions_hitl_gate.md) for the human-review gate.
