# HITL Approval

> **Notebook:** [`notebooks/subgraphs/09_hitl_approval.ipynb`](../../notebooks/subgraphs/09_hitl_approval.ipynb) · **Example:** [`examples/subgraphs/09_hitl_approval.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/09_hitl_approval.py)

This article covers prismal's Human-in-the-Loop (HITL) approval gate: `prismal.agents.subgraphs.gates` provides `seed_hitl_metadata()`, `human_approval_node()` and `hitl_gate()`, which together pause a LangGraph workflow at a critical decision point using LangGraph's `interrupt()`, persist the full state in a checkpointer, and resume only when a human replies with `Command(resume={"action": ...})`. The demo pipeline drafts an AI-deployment proposal, assesses its risk, and — for MEDIUM/HIGH risk — suspends until a reviewer approves, rejects, or requests changes.

Use this pattern whenever an agent is about to do something irreversible or high-stakes: production deployments, auto-merging PRs, actions touching PII or confidential data. Because the interrupt is checkpoint-backed, the pause can last seconds or days — the graph resumes exactly where it stopped.

## What it demonstrates

- Suspending a graph mid-run with `interrupt()` and resuming it with `graph.invoke(Command(resume=decision), config)` on the same `thread_id`.
- Why a **checkpointer is mandatory**: `interrupt()` persists state via the checkpointer (`MemorySaver` in the demo; SQLite/Postgres in production).
- Three-way routing on the human decision: `approve` / `reject` / `request_changes` (the last loops back through a revision handler, bounded at 3 iterations).
- Risk-based gating: LOW-risk proposals skip HITL entirely; MEDIUM/HIGH require a human.
- CI/CD bypass via `PRISMAL_HITL_ENABLED=false` or a `bypass_condition` callable — no interrupt is ever raised.

## Pipeline topology

```
proposal_writer ─► risk_assessor ─┬─ risk=LOW ──────────────► finalizer
                                  └─ risk=MEDIUM/HIGH
                                        │
                                  approval_seed        (seeds _hitl_* metadata)
                                        │
                                  human_approval       interrupt() ⏸  ... Command(resume=...)
                                        │
                                    hitl_gate ─┬─ approve ────────► finalizer ─► END
                                               ├─ reject ─────────► END
                                               └─ request_changes ► revision_handler ─► resubmit (≤ 3 iter)
```

## How it works

### Dataset: five deployment proposals

The notebook models an AI safety committee reviewing five proposals at different risk levels — from swapping the production model (HIGH: no canary release) and storing PII in long-term memory (HIGH: GDPR review pending) down to a prompt-safety update (LOW: trivially reversible). Each `Proposal` dataclass carries `risk_level`, `risk_reasons`, `expected_impact` and a `rollback_plan`, which is exactly the payload a reviewer needs to decide.

### The simulated pipeline: gate semantics without LangGraph

`run_hitl_pipeline()` reproduces the routing logic in plain Python so the control flow is easy to follow. The core loop shows the three-way decision and the bounded revision cycle:

```python
if decision.action == "approve":
    finalizer(current_proposal, decision.action, decision.modifications)
    return "approved"

if decision.action == "reject":
    return "rejected"

if decision.action == "request_changes":
    current_proposal = revision_handler(current_proposal, decision.modifications)
    iteration += 1
    if iteration >= MAX_ITERATIONS:
        return "max_iterations"   # auto-reject after 3 attempts
    continue                       # resubmit the revised proposal
```

LOW-risk proposals and `bypass=True` (CI/CD mode) short-circuit straight to the finalizer without any interrupt. The notebook runs this as an interactive demo (you type the decision), a scripted batch demo, and a bypass demo.

### The real thing: framework gates wired into a StateGraph

Demo 4 builds an actual LangGraph graph from the framework primitives. `seed_hitl_metadata()` writes `_hitl_artifact_field` / `_hitl_risk_level` into `state["metadata"]` so `human_approval_node` knows which artifact to surface in the interrupt payload; `hitl_gate()` is the conditional edge that routes on the recorded action:

```python
from prismal.agents.subgraphs.gates import hitl_gate, human_approval_node, seed_hitl_metadata

_seed_node = seed_hitl_metadata(artifact_field="artifact", risk_level="HIGH")
_gate = hitl_gate(artifact_field="artifact", on_approve="finalizer",
                  on_reject="rejected", risk_level="HIGH")

builder = StateGraph(ReviewState)
builder.add_node("approval_seed", _seed_node)
builder.add_node("human_approval", human_approval_node)
# ... writer, risk, finalizer, rejected nodes and linear edges ...
builder.add_conditional_edges("human_approval", _gate)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
```

Inside `human_approval_node`, `decision = interrupt(payload)` blocks until the resume arrives; the node validates the action (anything unrecognized defaults to `reject`), merges `modifications` into metadata on `request_changes`, and records `_hitl_last_action` for the gate to read.

### The interrupt/resume lifecycle

The two-invocation pattern is the heart of HITL — same `config` (thread), two calls:

```python
config = {"configurable": {"thread_id": "demo-hitl-001"}}

graph.invoke(initial_state, config)          # 1st call: suspends at human_approval
snapshot = graph.get_state(config)
print(snapshot.next)                          # ('human_approval',) — paused, state checkpointed

decision = {"action": "request_changes",
            "modifications": {"rollback_plan": "Canary 5%→25%→100% with auto-rollback"}}
graph.invoke(Command(resume=decision), config)  # 2nd call: resumes from the checkpoint
```

In this simplified demo `request_changes` routes to the `on_reject` target; in the real `dev_pipeline` that target is the developer node, so the workflow regenerates and re-submits. A second thread (`demo-hitl-002`) shows the `approve` path reaching the finalizer with `result == "DEPLOYED"`.

### Bypass for CI/CD

Two escape hatches skip the human entirely, both logged at WARNING: the settings flag (`PRISMAL_HITL_ENABLED=false` routes every gate straight to `on_approve`) and a per-gate callable (`hitl_gate(..., bypass_condition=lambda _: True)`) for tests or low-risk skips.

## Key API

| Symbol | Role |
|---|---|
| `human_approval_node(state)` | Async node: raises `interrupt(payload)`, validates the resumed decision, records `_hitl_last_action`, merges modifications |
| `hitl_gate(artifact_field, on_approve, on_reject, risk_level, bypass_condition)` | Conditional-edge factory: routes `approve → on_approve`, `reject`/`request_changes → on_reject`; honours config/callback bypass |
| `seed_hitl_metadata(artifact_field, risk_level)` | Node factory: seeds `_hitl_artifact_field` / `_hitl_risk_level`, clears stale actions — wire it just before `human_approval` |
| `interrupt(payload)` / `Command(resume=...)` | LangGraph primitives for pause and resume (from `prismal.langgraph` / `langgraph.types`) |
| `graph.compile(checkpointer=..., interrupt_before=[...])` | A checkpointer is required — interrupt state must be persisted to resume |
| `metadata["_hitl_last_action"]` | Where the human's action is recorded for the gate to route on |
| `settings.hitl_enabled` | `PRISMAL_HITL_ENABLED=false` → automatic approval (CI/CD mode) |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/09_hitl_approval.ipynb
uv run python examples/subgraphs/09_hitl_approval.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` for real mode; the simulated demos and the LangGraph interrupt demo run offline.

## Related

- [Dev Pipeline](03_dev_pipeline.md) — the production consumer of `hitl_gate`, looping rejected code back to the developer node.
- [Data ETL](05_data_etl.md) — a validation gate that blocks bad data automatically instead of asking a human.
- [Code Review](04_code_review.md) — automated review scores that often feed a HITL decision.
- [Debate Consensus](08_debate_consensus.md) — machine consensus as the alternative when a human gate is too slow.
