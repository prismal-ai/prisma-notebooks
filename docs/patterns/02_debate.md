# Debate / Society of Mind

> **Notebook:** [`notebooks/patterns/02_debate.ipynb`](../../notebooks/patterns/02_debate.ipynb) · **Example:** [`examples/patterns/02_debate.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/02_debate.py)

The Debate pattern (`prismal.agents.patterns.debate`, SPEC-PAT-002) answers a question by making N agents with distinct roles argue about it. Round 1 produces independent positions; in every subsequent round each agent sees the prior round's positions and refines its own — conceding valid points, pressing disagreements. A synthesis step (an LLM moderator by default) then reconciles the final positions into one consensus answer, and a Jaccard token-set similarity quantifies how much the agents actually agreed.

This is the pattern to reach for when a question has no single verifiable answer — ambiguous claims, trade-off decisions, ethics. The notebook uses **BoolQ-style controversial questions plus AI ethics dilemmas** (embedded, no download): should AI make hiring decisions unsupervised, do LLMs pose existential risk, is monetizing user data for training ethical, should frontier models be open source. Each topic gets its own cast of roles, from a classic proponent/opponent/neutral triad to a four-way panel of techno-optimist, existential pessimist, pragmatist, and ethicist.

## What it demonstrates

- Running a complete multi-agent debate with one call to `debate_round()` — agents, rounds, roles, and synthesis are all parameters, not code you write.
- Custom role casts per topic (e.g. `company_advocate` / `user_advocate` / `regulator` for a data-privacy question) and how role framing shapes positions.
- The `moderator` synthesis strategy: a neutral LLM integrates the stronger points instead of picking a winner.
- Interpreting `agreement_score` (pairwise Jaccard over final-round positions) as a coarse, embeddings-free consensus signal, plus surfacing `dissenting_views`.
- Comparing consensus across topics: which dilemmas converge and which stay contested.
- Fault tolerance for free: a single failing agent is skipped and logged, so one bad LLM call does not sink the debate.

## How it works

### Topics as data

Each debate is declared as a dict — question, cast, and round count — so the same runner handles all four:

```python
DEBATE_TOPICS = [
    {
        "query": "Should artificial intelligence be used to make hiring "
                 "decisions without human oversight?",
        "category": "ai_ethics",
        "n_agents": 3,
        "roles": ["proponent", "opponent", "neutral"],
        "n_rounds": 2,
    },
    {
        "query": "Do large language models (LLMs) pose an existential risk "
                 "to humanity within the next 20 years?",
        "category": "existential_risk",
        "n_agents": 4,
        "roles": ["techno_optimist", "existential_pessimist", "pragmatist", "ethicist"],
        "n_rounds": 3,
    },
    # ... data_privacy and open_source_AI topics
]
```

Roles are free-form labels: the pattern injects them into its internal position prompts ("You are the {role} in a structured debate..."), so a `regulator` and a `company_advocate` genuinely argue from different corners. If you pass `roles`, its length must equal `n_agents`; pass `None` and you get the default proponent/opponent/neutral triad (with `analyst_N` filling any overflow).

### One call runs the whole debate

```python
async def run_debate(topic: dict) -> DebateResult:
    return await debate_round(
        query=topic["query"],
        state={},  # opaque state — reserved for future extensions
        n_agents=topic["n_agents"],
        n_rounds=topic["n_rounds"],
        roles=topic["roles"],
        synthesis_strategy="moderator",
    )
```

Internally, round 1 asks each agent for an independent stance (agents are told *not* to acknowledge each other); rounds 2+ show each agent the previous round's positions and ask for a refined stance. Every prompt that carries the user's query or another agent's text is built through `SecurePromptBuilder`, so debate content is isolated from prompt templates — the framework's rule that user-controlled text is never f-stringed into a prompt applies inside patterns too. A single failing agent is logged and skipped; `DebateError` is raised only if *all* agents fail.

`synthesis_strategy="moderator"` sends the final-round positions to a neutral LLM moderator that must integrate the stronger points without copying any position verbatim. The alternatives are `majority_vote` (most frequent identical position wins — useful for short categorical answers) and `weighted` (a moderator variant told the per-position weights).

### Reading the result

`DebateResult` keeps the full transcript, so the notebook can replay the debate round by round before showing the verdict:

```python
for round_no in range(1, topic["n_rounds"] + 1):
    for pos in [p for p in result.positions if p.round == round_no]:
        print(f"[{pos.role}] {pos.content[:120]}...")

print(result.consensus)
print(f"Agreement (Jaccard): {result.agreement_score:.3f}")
```

Each `DebatePosition` records `agent_id`, `role`, `content`, and its 1-indexed `round`. The `agreement_score` is the mean pairwise Jaccard similarity over the *final* round's token sets — 1.0 means token-identical positions, 0.0 means no shared vocabulary. It is deliberately coarse: a cheap consensus signal that needs no embeddings infrastructure. The notebook buckets it (> 0.6 high consensus, > 0.3 moderate, otherwise high divergence) and prints `dissenting_views` — positions whose similarity to the consensus falls below 0.5, only surfaced when overall agreement was low.

### Comparing topics

The final cells tabulate all four debates and pick out the extremes:

```python
best = max(results, key=lambda x: x[1].agreement_score)
worst = min(results, key=lambda x: x[1].agreement_score)
print(f"Highest agreement : {best[0]['category']} ({best[1].agreement_score:.3f})")
print(f"Largest dispute   : {worst[0]['category']} ({worst[1].agreement_score:.3f})")
```

The interesting observation is usually that agreement tracks the question, not the cast size: pragmatic questions with shared vocabulary converge, while value-laden ones (existential risk, open-sourcing frontier models) stay divergent even after three rounds — which is precisely the signal you want before trusting a synthesized consensus.

## Key API

| Symbol | Role |
|---|---|
| `debate_round()` | Async entry point: runs N agents × M rounds, synthesizes consensus, returns a `DebateResult`; raises `ValueError` on bad counts, `DebateError` if every agent fails. |
| `DebateResult` | Dataclass: `consensus`, `agreement_score`, `positions` (all rounds), `dissenting_views`, `rounds_completed`. |
| `DebatePosition` | One agent's stance in one round: `agent_id`, `role`, `content`, `round`. |
| `pairwise_jaccard()` | Mean Jaccard similarity over token sets — the metric behind `agreement_score`, importable on its own. |
| `synthesis_strategy` | `"moderator"` (LLM reconciliation, default) · `"majority_vote"` · `"weighted"`. |

The pattern resolves its LLM internally via `ProviderRegistry` (optionally pass `settings=`), and accepts a `budget_guard_fn` that can stop additional rounds under Budget Governance.

## Run it

```bash
uv run jupyter lab notebooks/patterns/02_debate.ipynb
# or the script version:
uv run python examples/patterns/02_debate.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — every agent turn and the moderator are real LLM calls.

## Related

- [Tree of Thoughts (ToT)](01_tree_of_thoughts.md) — explores alternatives by tree search instead of multi-agent argument.
- [Debate Consensus subgraph](../subgraphs/08_debate_consensus.md) — the same idea packaged as a LangGraph pipeline (proponent → opponent → moderator → consensus).
- [Kokoro Deliberation](../kokoro_deliberation.md) — persona-driven deliberation where Markdown-authored souls argue toward agreement and a judge decides.
