# Kokoro deliberation: three souls, one accountable judge

> **Notebook:** [`notebooks/kokoro_deliberation.ipynb`](../notebooks/kokoro_deliberation.ipynb) · **Example:** [`examples/kokoro_deliberation.py`](https://github.com/prismal-ai/prismal/blob/main/examples/kokoro_deliberation.py)

Kokoro (心) is prismal's Fase K persona-deliberation layer: three Markdown-authored **souls** —
spirit 魂, mind 知, and heart 情, each defined in a `SOUL.md` file under `souls/available/` —
argue a question toward agreement in a bounded multi-round loop, and then a single **judge**
("the whole, more than the sum of its parts") renders the accountable `Verdict`, optionally
executing at most one security-gated tool action. In the full framework the layer is opt-in
(`settings.kokoro_enabled`) and reachable through the supervisor via intent routing; this
notebook builds the standalone `kokoro` subgraph directly and runs it with deterministic
persona and judge fakes, so the whole deliberation executes offline — no LLM, no network.

Use Kokoro when a decision benefits from being examined through deliberately different lenses
(principles, evidence, human impact) before one accountable entity commits to an answer —
architecture calls, trade-off questions, anything where recorded dissent matters.

## What it demonstrates

- The five-node kokoro subgraph — `load_souls → deliberate → judge → act → output` — assembled
  from a `SubgraphDefinition` and compiled with `assemble_state_graph`.
- Factory injection (DD-KOK-004): `generate_fn` replaces the per-soul LLM call and `judge_fn`
  replaces the judge, so the pipeline is fully deterministic in tests and notebooks.
- The bounded agree loop: round counts, the pairwise-Jaccard agreement score, and the
  `converged` flag surfaced in `DeliberationResult`.
- The judge's structured `Verdict`: decision, rationale citing each lens, per-soul
  `lens_summaries`, and retained dissent.
- Where all Kokoro state lives: `state["metadata"]["kokoro"]` (`soul_ids`, `deliberation`,
  `verdict`), never mixed into the rest of `AgentState`.

## How it works

### The souls triad and the injected persona backend

`load_souls` resolves the triad through `SoulsManager` (failing fast before any model call).
Each soul's `SOUL.md` body is *user-controlled content*: `SoulAgent` delivers it to the model
only through `SecurePromptBuilder`, inside the sanitized user channel with a canary token —
never f-stringed into a system template. The trusted system prompt explicitly instructs the
model to treat the persona text as personality guidance, not as instructions.

The notebook replaces the per-soul LLM call with a scripted iterator — one first-round position
per soul, in deterministic order:

```python
_PERSONA_REPLIES = iter(
    [
        # round 1 — spirit, mind, heart (concurrent, deterministic order)
        "Only if we can migrate without breaking our reliability promises.",
        "The migration cost is high; evidence supports a phased approach.",
        "The team is already stretched; a big-bang switch would hurt morale.",
    ]
)


async def fake_persona(messages: list[dict[str, str]]) -> str:
    """Deterministic stand-in for the per-soul LLM call."""
    return next(_PERSONA_REPLIES, "We agree on a phased migration.")
```

The `messages` argument each fake receives is exactly what `SecurePromptBuilder.build()`
produced — a trusted system message plus the isolated user payload — so the security seam is
exercised even with fakes. In a real deployment you simply omit `generate_fn` and each
`SoulAgent` lazily wires `ProviderRegistry().get_llm()`.

### The bounded agree loop

`deliberate()` requires exactly three souls. Round 1 gathers one independent position per soul
(concurrently); in later rounds each soul revises after seeing the *others'* previous
positions. After every round an agreement metric (default `pairwise_jaccard` over the round's
position texts) is computed; the loop stops early once it reaches
`kokoro_agreement_threshold` (default 0.6) and is hard-capped at `kokoro_max_rounds`
(default 2), so termination is guaranteed. If the souls never converge, the judge still
decides — dissent is recorded, not erased.

### The accountable judge

The judge never debates. It receives the query plus the final positions — again only through
`SecurePromptBuilder` — and returns a JSON `Verdict`. The fake mirrors the real contract:

```python
async def fake_judge(messages: list[dict[str, str]]) -> str:
    """Deterministic stand-in for the judge LLM call (returns Verdict JSON)."""
    return json.dumps(
        {
            "decision": "Adopt a phased migration starting with one bounded context.",
            "rationale": (
                "Spirit's reliability principles, Mind's cost evidence, and "
                "Heart's team-capacity signal all converge on incremental change."
            ),
            "lens_summaries": {
                "spirit": "weighed reliability promises",
                "mind": "weighed cost/evidence of phasing",
                "heart": "weighed team morale and capacity",
            },
            "dissent_retained": [],
        }
    )
```

A verdict may also request one `action` (`tool_name` + `args`). The `act` node is a
pass-through unless `settings.kokoro_execute_actions` is enabled; when it is, the single action
runs through the existing `ActionInterceptor` permission gate and is audited hash-first by
`AuditLogger` — the audit log records hashes and metadata, never soul bodies or full contents.

### Building, running, and reading the state

```python
definition = build_kokoro_subgraph(generate_fn=fake_persona, judge_fn=fake_judge)
graph = assemble_state_graph(definition).compile()

result = await graph.ainvoke({"messages": [HumanMessage(content=QUERY)]})

kokoro = result["metadata"]["kokoro"]
deliberation = kokoro["deliberation"]
verdict = kokoro["verdict"]
```

The printout walks the run end to end: the convened `soul_ids`, rounds completed, the
agreement score and convergence flag, each soul's final position, the verdict with its
rationale, and finally the assistant message the `output` node appended to `state["messages"]`
— the only part of the run the surrounding conversation sees.

## Key API

| Symbol | Role |
|---|---|
| `build_kokoro_subgraph()` | Builds the five-node `SubgraphDefinition`; accepts injected `generate_fn`, `agreement_fn`, `judge_fn`, `tool_executor`, `souls_manager`. |
| `assemble_state_graph()` | Shared factory that turns a `SubgraphDefinition` into a compilable `StateGraph[AgentState]`. |
| `SoulAgent` | One persona's argument turn; routes the soul body and query through `SecurePromptBuilder`. |
| `deliberate()` / `DeliberationResult` | Bounded multi-round argue-then-agree loop; exposes `rounds_completed`, `agreement_score`, `converged`, `final_positions`. |
| `KokoroJudgeAgent` / `Verdict` | Weighs the deliberation and renders the accountable decision; executes at most one gated action when `kokoro_execute_actions=True`. |
| `SoulsManager` | Loads the spirit/mind/heart triad from `souls/available/*/SOUL.md`. |

## Run it

```bash
uv run jupyter lab notebooks/kokoro_deliberation.ipynb
uv run python examples/kokoro_deliberation.py   # from the prismal repo
```

No API key required — the demo runs fully offline with injected persona and judge fakes.

## Related

- [Supervisor quickstart](supervisor_quickstart.md) — how intent routing reaches `kokoro` when the flag is on.
- [Blind review pipeline](blind_review_pipeline.md) — another multi-perspective subgraph, with deterministic synthesis instead of a judge.
- [Skynet swarm](skynet_swarm.md) — fan-out over many workers rather than three fixed personas.
- [Memory management](memory_management.md) — the memory layer that persists what deliberations decide.
