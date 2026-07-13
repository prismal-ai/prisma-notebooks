# Blind review pipeline: spec, implement, two independent reviewers

> **Notebook:** [`notebooks/blind_review_pipeline.ipynb`](../notebooks/blind_review_pipeline.ipynb) · **Example:** [`examples/blind_review_pipeline.py`](https://github.com/prismal-ai/prismal/blob/main/examples/blind_review_pipeline.py)

The blind review pipeline (Phase BRP) is a quality-control subgraph in
`prismal/agents/subgraphs/`: a spec agent turns a goal into a specification, an implementer
produces an artifact against it, and **two blind reviewers** — each seeing only the
`(spec, artifact)` pair, never the other reviewer's verdict — score it independently. A
deterministic, LLM-free synthesis then merges the two verdicts conservatively, a score gate
either loops the artifact back to the implementer (bounded correction loop) or forwards it to a
human-in-the-loop approval gate. The blindness contract is what makes the pattern valuable:
because neither reviewer can anchor on the other, agreement between them is a genuine signal.

This notebook builds the subgraph with deterministic role backends — every LLM seat is an
injected callable — and disables the HITL gate so a single `ainvoke` runs to `END` offline. It
is the reference walkthrough for gated, multi-reviewer generation flows in prismal.

## What it demonstrates

- The pipeline topology: `spec_agent → implementer → reviewer_a → reviewer_b → synthesis`,
  with a score gate that loops failures back to the implementer and routes passes toward HITL.
- Factory injection for all four roles (`spec_fn`, `implementer_fn`, `reviewer_a_fn`,
  `reviewer_b_fn`), so the run is deterministic and needs no provider key.
- Reviewer blindness: each reviewer's contract is the narrow `(spec, artifact)` input — the
  independence is structural, enforced by the input signature and guard tests, not by prompt
  discipline.
- Deterministic synthesis: de-duplicated union of issues, conservative `min` score, re-derived
  approval, and an `agreement` flag comparing the two verdicts.
- How HITL is toggled: `PRISMAL_HITL_ENABLED=false` for an uninterrupted demo, versus a real
  deployment resuming the `interrupt()` with `Command(resume={"action": "approve"})`.

## How it works

### Turning off the human gate for the demo

`human_approval_node` always raises LangGraph's `interrupt()`, so with HITL enabled a passing
run pauses and waits for a resume command. The notebook flips the setting before importing
anything from prismal:

```python
# Disable the human-in-the-loop gate so the demo completes without an interrupt.
# (A real deployment leaves PRISMAL_HITL_ENABLED=true and resumes the interrupt
# with Command(resume={"action": "approve"}).)
os.environ.setdefault("PRISMAL_HITL_ENABLED", "false")
```

When the flag is off, the synthesis gate routes a passing score straight to `END` instead of
entering the approval sub-flow (the enablement check must gate *entry*, since the approval node
would otherwise still pause the run).

### The injected role backends

Each seat in the pipeline is an async callable with a role-specific signature. The spec fake
returns a fixed specification; the implementer fake shows the correction-loop seam — it
receives `prior_issues` when the score gate has sent the artifact back for revision:

```python
async def fake_implement(spec: str, prior_issues: list[CodeIssue] | None) -> str:
    """Deterministic stand-in for the implementer agent's LLM call."""
    note = " (revised to address reviewer issues)" if prior_issues else ""
    return f"def parse_csv(path):{note}\n    ...  # implementation matching the spec"
```

The reviewers return structured `CodeReviewReport` objects (reused from the `code_review`
subgraph). Reviewer A plays a correctness lens and approves cleanly; reviewer B plays a
robustness lens and flags a low-severity edge case while still approving:

```python
async def fake_reviewer_b(spec: str, artifact: str) -> CodeReviewReport:
    """Blind reviewer B — a robustness lens. Flags a minor edge case."""
    return CodeReviewReport(
        issues=[
            CodeIssue(
                severity="low",
                category="logic",
                description="empty-file case not explicitly handled",
                file="parse_csv.py",
                line=1,
                suggestion="return [] for an empty file",
            )
        ],
        summary="minor edge case",
        score=0.85,
        approved=True,
    )
```

Note what the reviewers *cannot* see: no conversation history, no other verdict — just spec and
artifact. (In the framework the reviewers run sequentially rather than fanned out, because both
write the un-reduced `metadata` channel of the shared `AgentState`; blindness is preserved by
the input contract, not by concurrency.)

### Build, run, and read the state

The builder wires the fakes into a `SubgraphDefinition`; `assemble_state_graph` compiles it.
A checkpointer and `thread_id` are required because the HITL machinery is interrupt-capable:

```python
definition = build_blind_review_pipeline_subgraph(
    spec_fn=fake_spec,
    implementer_fn=fake_implement,
    reviewer_a_fn=fake_reviewer_a,
    reviewer_b_fn=fake_reviewer_b,
)
graph = assemble_state_graph(definition).compile(checkpointer=MemorySaver())

result = await graph.ainvoke(
    {"messages": [HumanMessage(content=GOAL)], "metadata": {}},
    config={"configurable": {"thread_id": "demo"}},
)

br = result["metadata"]["blind_review"]
synthesis = br["synthesis"]
```

Everything the pipeline produced lives under `state["metadata"]["blind_review"]`: the
`spec_artifact`, the `implementation_artifact`, both raw verdicts
(`reviewer_a_verdict` / `reviewer_b_verdict`), and the merged `synthesis`.

### Deterministic synthesis and the gates

`synthesize_verdicts` is pure code — no third LLM opinion is consulted:

- `report.issues` — the de-duplicated union of both issue lists (dedupe key:
  `(file, line, category, description)`).
- `report.score` — `min(a.score, b.score)`, the conservative choice.
- `report.approved` — re-derived as `score >= blind_review_approval_threshold`.
- `agreement` — whether the two reviewers' `approved` flags match.

The score gate then reads `blind_review.synthesis.report.score`: below the threshold it routes
back to `implementer` (passing the merged issues as `prior_issues`), bounded by
`blind_review_max_iterations`; at or above it, the run proceeds to the HITL trio — or directly
to `END` here, since HITL is disabled. In this demo the merged score is `min(0.9, 0.85) = 0.85`,
both reviewers approved (`agreement=True`), and the single merged issue is reviewer B's
empty-file edge case, so the run passes on the first iteration.

## Key API

| Symbol | Role |
|---|---|
| `build_blind_review_pipeline_subgraph()` | Builds the `SubgraphDefinition` with injectable `spec_fn`, `implementer_fn`, `reviewer_a_fn`, `reviewer_b_fn`, `synthesize_fn`. |
| `assemble_state_graph()` | Shared factory that turns a `SubgraphDefinition` into a compilable `StateGraph[AgentState]`. |
| `CodeReviewReport` / `CodeIssue` | Structured reviewer verdicts and findings, reused from the `code_review` subgraph. |
| `synthesize_verdicts()` / `SynthesisResult` | Deterministic, LLM-free merge: issue union, `min` score, re-derived approval, `agreement` flag. |
| `score_gate` / `hitl_gate` / `human_approval_node` | Reused gate primitives: bounded correction loop and the `interrupt()`-based human approval. |
| `MemorySaver` | In-memory LangGraph checkpointer, required because the pipeline is interrupt-capable. |

## Run it

```bash
uv run jupyter lab notebooks/blind_review_pipeline.ipynb
uv run python examples/blind_review_pipeline.py   # from the prismal repo
```

No API key required — all four roles are injected deterministic fakes and the run stays offline.

## Related

- [Kokoro deliberation](kokoro_deliberation.md) — multiple perspectives resolved by a judge instead of a deterministic merge.
- [Supervisor quickstart](supervisor_quickstart.md) — the state machine subgraphs like this plug into.
- [Agent evaluation](agent_eval.md) — scoring agent outputs against expectations.
- [Visualize graphs](visualize_graphs.md) — rendering subgraph topologies like this one as Mermaid diagrams.
