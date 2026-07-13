# Debate Consensus

> **Notebook:** [`notebooks/subgraphs/08_debate_consensus.ipynb`](../../notebooks/subgraphs/08_debate_consensus.ipynb) · **Example:** [`examples/subgraphs/08_debate_consensus.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/08_debate_consensus.py)

The `debate_consensus` subgraph synthesizes a balanced position on a contested thesis by running one structured debate round through four fixed roles: **proponent → opponent → moderator → consensus**. The proponent builds the strongest argument in favor, the opponent the strongest argument against, the moderator evaluates both and surfaces agreement/tension points, and the terminal consensus node synthesizes a final answer plus a quantitative Jaccard agreement score. It reuses the `DebatePosition` / `pairwise_jaccard` primitives from `prismal.agents.patterns.debate` — the subgraph is the structured, single-round packaging of that pattern.

Use it for complex binary decisions (adopt X vs Y), risk analysis, balanced content on controversial topics, or design-proposal validation. The notebook debates four AI-policy theses (regulation, open-source LLMs, algorithmic hiring, generative-AI copyright) chosen because both sides have legitimate arguments.

## What it demonstrates

- A 4-role debate pipeline where each node appends a `DebatePosition` to `state["metadata"]["debate_consensus"]["positions"]`.
- Quantifying agreement with `pairwise_jaccard()` — 0.0 means fully opposed positions, 1.0 identical.
- A moderator stage that separates points of convergence from unresolved tensions before synthesis.
- The trade-off between this fixed 2-role, 1-round subgraph and the flexible N-agent, M-round `debate.py` pattern.
- The standard build pattern: `build_debate_consensus_subgraph()` returns a `SubgraphDefinition`, compiled via `assemble_state_graph(defn).compile()`.

## Pipeline topology

```
proponent ──► opponent ──► moderator ──► consensus ──► END
 (argue FOR)  (argue AGAINST)  (agreement +      (synthesis +
                                tension points)   Jaccard score)
```

One structured round; for multi-round rebuttal loops use `patterns/debate.py::debate_round()` with `n_rounds > 1`.

## How it works

### Dataset: theses with expected agreement levels

Each topic carries an `expected_agreement` label so the notebook can check whether the measured Jaccard score matches intuition:

```python
DEBATE_TOPICS = [
    {
        "id": "DEB-001",
        "thesis": "Governments should strictly regulate AI development through "
        "mandatory licensing and independent audits before deployment.",
        "domain": "AI Regulation",
        "context": "Reference: EU AI Act (2024), GPAI governance proposals, ...",
        "expected_agreement": "medium",  # clear points of agreement + disagreement
    },
    # DEB-002: Open Source AI (low) · DEB-003: Algorithmic Hiring (high)
    # DEB-004: AI Copyright (medium)
]
```

A `DEBATE_ARGUMENTS` dict provides pre-elaborated proponent/opponent arguments plus curated `agreement_points` and `tension_points`, so the demo runs deterministically without an LLM.

### Moderator: measuring how far apart the positions are

The moderator computes the Jaccard similarity between the two argument sets — the lower the score, the more divergent the debate. When prismal is importable it uses the real primitive:

```python
def simulate_moderator(args: dict) -> dict:
    proponent_args = args["proponent"]
    opponent_args = args["opponent"]
    if DEBATE_PRIMITIVES_AVAILABLE:
        jaccard = pairwise_jaccard(proponent_args, opponent_args)
    else:
        jaccard = simple_jaccard(" ".join(proponent_args), " ".join(opponent_args))
    return {
        "jaccard_agreement": jaccard,
        "agreement_points": args.get("agreement_points", []),
        "tension_points": args.get("tension_points", []),
        ...
    }
```

The consensus stage then maps the score into a stance: above ~0.15 signals partial agreement, below ~0.08 a polarized debate with no clear consensus.

### Real mode: build, register, compile, invoke

The real subgraph follows the standard pattern — the builder returns a `SubgraphDefinition` (4 nodes, linear edges, entry point `proponent`) and `assemble_state_graph(defn).compile()` yields the runnable graph:

```python
await register_debate_consensus()
subgraph = build_debate_consensus_subgraph()
graph = assemble_state_graph(subgraph).compile()

state = create_initial_state(session_id="nb-debate-consensus")
state["messages"] = [HumanMessage(content=f"Debate on the following thesis:\n{topic['thesis']}\n\nContext: {topic['context']}")]
state["metadata"] = {"debate_consensus": {"thesis": topic["thesis"], "domain": topic["domain"]}}

config = {"configurable": {"thread_id": f"debate_{topic['id']}_001"}}
final_state = await graph.ainvoke(state, config=config)
```

The terminal consensus node appends an `AIMessage` with the synthesis and records `consensus` and `jaccard_score` under `metadata["debate_consensus"]`, computed as `pairwise_jaccard([p.content for p in positions])` over all collected positions.

### Subgraph vs. the full debate pattern

The subgraph fixes 2 roles and 1 round for a structured, repeatable flow. For N agents and M rebuttal rounds, drop down to the pattern:

```python
from prismal.agents.patterns.debate import debate_round

result = await debate_round(
    query=thesis, state=state,
    n_agents=5,   # proponent, opponent, + 3 specialists
    n_rounds=3,   # rebuttal and counter-rebuttal
    roles=["economist", "ethicist", "engineer", "regulator", "citizen"],
    synthesis_strategy="moderator",
)
# result.agreement_score: average Jaccard across all positions
```

Avoid the subgraph for questions with an objectively correct answer — debate adds cost without adding signal there.

## Key API

| Symbol | Role |
|---|---|
| `build_debate_consensus_subgraph(llm, settings)` | Builds the 4-node `SubgraphDefinition` (entry point `proponent`, linear edges) |
| `register_debate_consensus(...)` | Idempotent install into the `SubgraphRegistry` singleton (async) |
| `assemble_state_graph(defn)` | Shared factory turning a `SubgraphDefinition` into a compilable `StateGraph` |
| `make_proponent_node` / `make_opponent_node` / `make_moderator_node` | Role node factories; each appends a `DebatePosition` to metadata |
| `make_consensus_node(llm)` | Terminal node: synthesis `AIMessage` + `jaccard_score` via `pairwise_jaccard` |
| `DebatePosition`, `pairwise_jaccard()` | Reused primitives from `prismal.agents.patterns.debate` |
| `metadata["debate_consensus"]` | State channel: `positions`, `consensus`, `jaccard_score` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/08_debate_consensus.ipynb
uv run python examples/subgraphs/08_debate_consensus.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` for real mode; the simulated debate runs offline.

## Related

- [Document Generation](07_document_generation.md) — sibling linear pipeline producing documents instead of verdicts.
- [HITL Approval](09_hitl_approval.md) — when the final call should come from a human, not a consensus node.
- [Code Review](04_code_review.md) — multiple reviewer perspectives merged into one report, the code-centric analogue.
- [Analysis Orchestrator](10_analysis_orchestrator.md) — routes analytical work across subgraphs like this one.
