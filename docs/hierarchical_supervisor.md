# Hierarchical supervision: domain supervisors and network routing

> **Notebook:** [`notebooks/hierarchical_supervisor.ipynb`](../notebooks/hierarchical_supervisor.ipynb) · **Example:** [`examples/hierarchical_supervisor.py`](https://github.com/prismal-ai/prismal/blob/main/examples/hierarchical_supervisor.py)

The flat supervisor loop routes every turn across all 26 specialists, which makes each routing
prompt large and each LLM decision harder. prismal offers two hierarchical layers on top of it.
The first is *domain supervision*: in hierarchical mode (`PRISMAL_HIERARCHICAL_MODE=true`) the
root supervisor routes to three domain orchestrators (research / engineering / analysis) instead
of 12+ leaf agents, and each domain supervisor — built by `make_domain_supervisor` — picks among
roughly three leaf agents, keeping every routing prompt focused and predictable. The second is
*network supervision*: `NetworkSupervisorAgent` delegates tasks to remote prismal nodes selected
by capability tag, falling back gracefully to the local graph when no node matches or the remote
call fails.

This notebook exercises both layers fully offline. The domain supervisor factory does not take an
injectable LLM, so the demo patches the module's `ProviderRegistry` with a deterministic fake to
walk through the three routing outcomes; the network supervisor's two I/O seams (the HTTP call and
the local graph run) are overridden in a subclass so its capability-selection and fallback logic
runs with no server and no LLM.

## What it demonstrates

- How `make_domain_supervisor(domain_name, agents, domain_description)` manufactures a scoped
  routing node with the same contract as the root supervisor.
- The three routing outcomes of a domain supervisor: a valid leaf-agent pick, the loop breaker
  (one specialist response per turn, no LLM call), and the invalid-response → `END` default that
  guarantees the graph always terminates.
- Per-domain isolated iteration counters kept under `state["metadata"]["domain_<name>"]`, so
  parallel domains never fight over shared loop state.
- Capability-based delegation with `NetworkSupervisorAgent.delegate()`: remote-first ordering and
  the graceful local-graph fallback when no node advertises the capability.
- How to fake external seams (routing LLM, HTTP, local graph) to test hierarchical routing
  deterministically.

## How it works

### Faking the routing LLM

`make_domain_supervisor` resolves its LLM internally via `ProviderRegistry().get_llm_with_fallback()`,
so the notebook substitutes the whole registry class with a stand-in whose `ainvoke` ignores the
prompt and returns one canned reply:

```python
class FakeRoutingLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(content=self._reply)


class FakeProviderRegistry:
    reply: str = "researcher"

    def get_llm_with_fallback(self) -> FakeRoutingLLM:
        return FakeRoutingLLM(type(self).reply)
```

Swapping `reply` between test cases makes each routing branch reachable on demand — exactly the
technique the framework's own unit tests use for supervisor logic.

### Demo 1 — the three routing outcomes

The factory returns an async LangGraph node function scoped to a domain and its allowed leaf
agents. The node name is derived from the domain (`research_supervisor` here):

```python
research_supervisor = make_domain_supervisor(
    domain_name="research",
    agents=["researcher", "rag_agent"],
    domain_description=(
        "Handles web research, document retrieval and knowledge-base "
        "questions. Routes to the researcher for open-web queries and "
        "to the rag_agent for indexed-document questions."
    ),
)
```

With `ProviderRegistry` patched, the demo runs three states through it:

```python
# (a) Fake LLM answers "researcher" -> valid route.
FakeProviderRegistry.reply = "researcher"
update = await research_supervisor(_fresh_state("hier-demo-a"))

# (b) Loop breaker: last message is a specialist's AIMessage.
state = _fresh_state("hier-demo-b")
state["messages"].append(AIMessage(content="Here are three papers ..."))
state["current_agent"] = "researcher"
update = await research_supervisor(state)

# (c) Invalid routing decision defaults to END (logged at WARNING).
FakeProviderRegistry.reply = "banana"
update = await research_supervisor(_fresh_state("hier-demo-c"))
```

Case (a) validates the reply against the domain's `frozenset(agents) | {"END"}` route set and sets
`next_agent="researcher"`. Case (b) never calls the LLM: because the most recent message is an
`AIMessage` and `current_agent` is not the domain supervisor itself, the loop breaker fires and
ends the turn — the same one-specialist-per-turn invariant the root supervisor enforces. Case (c)
shows the safety default: `"banana"` matches nothing, is logged at WARNING, and maps to `END`, so a
misbehaving routing LLM can never trap the graph in a loop. A hard cap of 8 iterations per domain
(tracked in the `metadata["domain_research"]["iteration_count"]` slot the demo prints) backs all of
this up by force-routing to `END` before touching the LLM again.

### Demo 2 — network delegation with faked I/O

`NetworkSupervisorAgent.delegate(task, capability, session_id)` picks the first enabled
`NetworkNode` whose `capabilities` list contains the requested tag, POSTs the task to that node's
`/api/v1/agent/chat` endpoint with a short-lived HS256 bearer JWT, and on any failure — or when no
node matches — runs the task on the local compiled graph instead. The demo subclasses it to
replace only the two I/O seams, inheriting `delegate()` unchanged:

```python
class OfflineNetworkSupervisor(NetworkSupervisorAgent):
    async def _call_remote(self, node, task, session_id):
        reply = f"[{node.name}] handled task: {task}"
        return {"messages": [AIMessage(content=reply)],
                "current_agent": "network_supervisor"}

    async def _run_local(self, task, session_id):
        return {"messages": [AIMessage(content=f"[local graph] handled task: {task}")],
                "current_agent": "network_supervisor"}
```

A two-node network (`research-node-eu` with `research`/`rag`, `coder-node-us` with `coding`) then
handles three tasks: a `research` task lands on the EU node, a `coding` task on the US node, and a
`quantum` task — a capability no node advertises — takes the local-graph fallback. In production
the node registry is loaded from `config/network_nodes.yaml` when no `nodes` list is passed.

## Key API

| Symbol | Role |
|---|---|
| `prismal.agents.domain_supervisor.make_domain_supervisor` | Factory producing a scoped async routing node (`<domain>_supervisor`) for a subset of leaf agents |
| `prismal.agents.network_supervisor.NetworkSupervisorAgent` | Delegates tasks to remote prismal nodes by capability, with local fallback |
| `prismal.agents.network_supervisor.NetworkNode` | Pydantic model for a remote node: `name`, `url`, `capabilities`, `enabled`, `timeout_seconds` |
| `NetworkSupervisorAgent.delegate` | Selects a node, calls it over HTTP with a bearer JWT, or falls back to the local graph |
| `prismal.agents.state.create_initial_state` | Fresh `AgentState` used as the base for each routing scenario |

## Run it

```bash
uv run jupyter lab notebooks/hierarchical_supervisor.ipynb
uv run python examples/hierarchical_supervisor.py   # from the prismal repo
```

No API key required — both demos run offline with injected fakes.

## Related

- [Supervisor quickstart](supervisor_quickstart.md) — the flat supervisor loop these layers build on
- [Parallel research fan-out](parallel_research_fanout.md) — parallelism within one graph instead of across nodes
- [Skynet swarm](skynet_swarm.md) — the map-reduce meta-supervisor for goal decomposition
- [Subagent spawner](subagent_spawner.md) — dynamic delegation to spawned sub-agents
