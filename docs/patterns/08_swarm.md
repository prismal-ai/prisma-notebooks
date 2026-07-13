# Swarm / Handoff

> **Notebook:** [`notebooks/patterns/08_swarm.ipynb`](../../notebooks/patterns/08_swarm.ipynb) · **Example:** [`examples/patterns/08_swarm.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/08_swarm.py)

The Swarm pattern lets specialized agents pass control directly to one another — no central supervisor deciding every hop. Each agent looks at the task, decides it belongs to a colleague, and hands off; the receiving agent may resolve the query or hand off again. This decentralised routing shines in support-desk scenarios where the *current* agent has the best local knowledge about who should act next, and where a supervisor round-trip per hop would add latency and cost.

prismal implements the handoff itself as a small, pure, heavily-guarded function: `swarm_handoff()`. It is deliberately boring — immutable state, an allow-list of valid targets, self-handoff rejection, and a complete audit trail — because in a system with no central authority, the *transfer mechanism* is where safety has to live. The notebook drives it with a customer-support scenario inspired by the ATIS dataset (~5,871 airline-travel utterances across 26 intents): five support tickets (order tracking, SQL/cohort analysis, a broken sales report, a production bug, sprint planning) are routed hop by hop through `researcher`, `coder`, `data_analyst`, `critic`, `file_manager`, and `planner`.

## What it demonstrates

- Decentralised routing: a keyword intent classifier plus a per-intent routing table decide each next hop locally — no supervisor node in the loop.
- `swarm_handoff()` mechanics: the new state carries `next_agent` and an appended `HandoffRecord` entry in `state["metadata"]["handoff_history"]`.
- The safety guarantees, exercised live: self-handoff raises `ValueError`, and a target outside `VALID_HANDOFF_TARGETS` is rejected.
- Anti-loop discipline at the application layer: a `max_hops` budget and a visited-agents set prevent ping-pong routing.
- Reconstructing an agent trajectory from the audit trail (from/to, reason, ISO timestamp, context snapshot) after the fact.

## How it works

### Local routing decisions

Nothing in the pattern mandates *how* an agent decides to hand off — that logic is yours. The notebook keeps it deterministic and inspectable: classify the ticket's intent from keywords, then walk a routing table of specialist sequences, skipping agents already visited (loop prevention) and anything not on the allow-list:

```python
routing_table: dict[str, list[str]] = {
    "technical_bug": ["coder", "critic"],
    "data_analysis": ["data_analyst", "file_manager"],
    "order_support": ["file_manager"],
    "planning": ["planner"],
    "coding": ["coder"],
    "general": [],
}

for target in target_sequence:
    if target not in agents_visited and target in VALID_HANDOFF_TARGETS:
        return target, reasons.get(target, f"Specialist in {target}")
return None  # current agent resolves the ticket
```

In production this classifier would be an LLM or a trained model — the handoff machinery does not change.

### Executing a handoff

Once an agent decides to delegate, the actual transfer is one call. Note the mandatory `reason` — every hop must be justified for the audit trail — and the explicit allow-list:

```python
state = await swarm_handoff(
    current_agent=current_agent,
    target_agent=target_agent,
    state=state,
    reason=reason,
    valid_targets=VALID_HANDOFF_TARGETS,
)
current_agent = state.get("next_agent", target_agent)
```

`swarm_handoff` never mutates its input: it returns a new state dict with a fresh `metadata` sub-dict, `next_agent` set to the target, and the handoff appended to `metadata["handoff_history"]`. The recorded snapshot copies only a few small state keys (`session_id`, `iteration_count`, `current_agent`, `task_plan`, `risk_score`) and deep-copies them so later state mutations can't retroactively corrupt the audit record — it never snapshots `messages` or other large blobs. Each successful hop also emits a `swarm_handoff_recorded` structlog event inside a `swarm.handoff` OTel span.

### Guardrails, demonstrated

A swarm with no supervisor needs its guarantees enforced at the primitive, and the notebook proves both failure modes fire:

```python
# Self-handoff → ValueError (an agent delegating to itself is almost always a bug)
await swarm_handoff(current_agent="coder", target_agent="coder",
                    state={"metadata": {}}, reason="self-handoff test")

# Off-list target → ValueError (allow-listing keeps the reachability graph tractable)
await swarm_handoff(current_agent="coder", target_agent="malicious_agent",
                    state={"metadata": {}}, reason="invalid target test",
                    valid_targets=VALID_HANDOFF_TARGETS)
```

The allow-list matters beyond typo protection: it means the set of agents reachable from any point in the swarm is a known, reviewable constant, not an emergent property of prompt text.

### Bounding the loop

Even with self-handoffs and off-list targets rejected at the primitive, a badly written routing policy could still bounce a ticket between two valid agents forever. The notebook closes that gap at the application layer with a hop budget and a visited set:

```python
hop = 0
max_hops = 5  # anti-loop safety limit
while hop < max_hops:
    handoff_decision = decide_handoff(current_agent, intent, state)
    if handoff_decision is None:
        break  # current agent resolves the ticket
    ...
    hop += 1
```

This division of labour is deliberate: the primitive enforces the invariants that must *never* be violated (immutability, allow-list, no self-handoff, audit), while termination policy — how many hops a ticket may take — stays in your hands.

### Reading the trail

After processing all five tickets, the notebook prints a routing summary (ticket, channel, intent, hop count) and then replays the full handoff history of the last ticket:

```python
history = last_state.get("metadata", {}).get("handoff_history", [])
for i, record in enumerate(history, 1):
    print(f"{i}. {record['from_agent']} → {record['to_agent']}")
    print(f"   Reason   : {record['reason']}")
    print(f"   Timestamp: {record['timestamp']}")
```

Because the history lives in ordinary state metadata, any downstream consumer (dashboard, audit job, evaluator) can reconstruct the trajectory without instrumenting the agents themselves.

## Key API

| Symbol | Role |
|---|---|
| `swarm_handoff(current_agent, target_agent, state, reason, valid_targets=None)` | Async pure function transferring control; returns a new state with `next_agent` set and the handoff appended to `metadata["handoff_history"]`. Raises `ValueError` on self-handoff or off-list targets. |
| `VALID_HANDOFF_TARGETS` | Default frozen allow-list: `coder`, `researcher`, `data_analyst`, `planner`, `file_manager`, `rag_agent`, `critic`. |
| `HandoffRecord` | Dataclass shape of one history entry: `from_agent`, `to_agent`, `reason`, `timestamp` (ISO-8601 UTC), `context_snapshot`. |
| `SwarmError` | Pattern-level exception type re-exported from `prismal.core.exceptions`. |

## Run it

```bash
uv run jupyter lab notebooks/patterns/08_swarm.ipynb
# or the script version:
uv run python examples/patterns/08_swarm.py   # from the prismal repo
```

Set an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — the routing demo itself is deterministic, but the notebook environment expects the standard provider configuration.

## Related

- [Mixture of Agents](05_mixture_of_agents.md) — the centralized alternative: parallel proposers plus an aggregator, rather than sequential decentralized handoffs.
- [Skynet swarm](../skynet_swarm.md) — prismal's supervised swarm: a meta-supervisor plans, fans out workers, and reduces — contrast with this pattern's supervisor-free transfers.
- [Debate consensus subgraph](../subgraphs/08_debate_consensus.md) — structured multi-agent turn-taking with a moderator instead of free handoffs.
- [Supervisor quickstart](../supervisor_quickstart.md) — the central SUPERVISOR architecture that swarm handoffs deliberately bypass.
