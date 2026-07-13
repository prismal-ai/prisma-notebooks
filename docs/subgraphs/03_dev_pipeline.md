# Dev Pipeline

> **Notebook:** [`notebooks/subgraphs/03_dev_pipeline.ipynb`](../../notebooks/subgraphs/03_dev_pipeline.ipynb) · **Example:** [`examples/subgraphs/03_dev_pipeline.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/03_dev_pipeline.py)

The Dev Pipeline subgraph models an agile software team as six agents: PO → Architect → Developer → UnitTest → QA → Reviewer. A GitHub-issue-style feature request goes in one end; user stories, an architecture, an implementation, tests, and a reviewed code artifact come out the other. Four gates keep quality honest: a test gate (failing tests send control back to the developer, max 3 retries), a review gate (reviewer score >= 0.8 or back to the developer), and a two-step HITL flow (an approval seed plus a human-approval interrupt) that can be bypassed entirely for CI/CD.

It is also the pipeline that showcases dynamic parallelism: after the developer, a module router uses LangGraph `Send()` to fan unit testing out across one worker per module, with an aggregator merging the results before the test gate.

## What it demonstrates

- The `register_dev_pipeline()` + `get_compiled_dev_pipeline()` convenience API shared by pipelines 01–03.
- Four gates: test gate, review gate, HITL seed, and HITL gate (with `hitl_enabled=False` bypass).
- Dynamic fan-out with `Send()` — `module_dispatcher` spawns one `dev_unit_tester` per module, converging on `dev_test_aggregator`.
- Bounded retry loops (`max_iterations=3`) as the anti-infinite-loop guard on both the test and review gates.
- Driving the pipeline from realistic GitHub issues and reading `iteration`, `review_result.score`, and `code_artifact` from `metadata["dev_pipeline"]`.

## Pipeline topology

```
po_agent ──► architect ──► developer ──► module_router
                              ▲              │  Send() per module
                              │       ┌──────┴───────────────┐
                              │  dev_unit_tester × N     unit_tester (sequential)
                              │       │                      │
                              │  dev_test_aggregator         │
                              │       └───────┬──────────────┘
                              │     [Test Gate: failing_tests empty? max 3]
                              ├──── NO ───────┤
                              │              YES
                              │               ▼
                              │        qa_agent ──► reviewer
                              │     [Review Gate: score >= 0.8? max 3]
                              ├──── NO ───────┤
                              │              YES
                              │               ▼
                              │   approval_seed ──► human_approval
                              │     [HITL Gate: bypass if hitl_enabled=False]
                              └── reject ─────┤
                                          approve
                                              ▼
                                             END
```

## How it works

### Registering and compiling the subgraph

As with the other classic pipelines, one idempotent registration call plus one accessor for the compiled graph:

```python
from prismal.agents.subgraphs.dev_pipeline.builder import (
    get_compiled_dev_pipeline,
    register_dev_pipeline,
)

await register_dev_pipeline()
compiled = await get_compiled_dev_pipeline()
```

### Turning an issue into a task

The dataset is three realistic GitHub issues (rate limiting middleware, async checkpoint context manager, a Typer CLI). Each is formatted into a single instruction that walks the six roles:

```python
def format_issue_as_task(issue: dict) -> str:
    return (
        f"Issue #{issue['id']}: {issue['title']}\n"
        f"Repository: {issue['repo']}\n"
        f"Priority: {issue['priority']} | Complexity: {issue['complexity']}\n"
        f"Labels: {', '.join(issue['labels'])}\n\n"
        f"Full description:\n{issue['description']}\n\n"
        f"Estimated time: {issue['estimated_hours']} hours\n\n"
        "Develop this feature following the full pipeline:\n"
        "1. PO: define user stories and acceptance criteria\n"
        "2. Architect: design architecture and interfaces\n"
        "3. Developer: implement the code in Python\n"
        "4. UnitTest: write unit tests with pytest\n"
        "5. QA: verify coverage and edge cases\n"
        "6. Reviewer: review quality, security and style"
    )
```

### Seeding state and invoking

The metadata pre-seeds the iteration counters and disables HITL so the demo runs unattended; `recursion_limit` is raised to 25 to leave room for the retry loops:

```python
state = create_initial_state(session_id="nb-dev-pipeline")
state["messages"] = [HumanMessage(content=format_issue_as_task(issue))]
state["metadata"] = {
    "issue_id": issue["id"],
    "priority": issue["priority"],
    "hitl_enabled": False,  # disable human approval in the example
    "dev_pipeline": {"iteration": 0, "review_attempts": 0},
}
config = {"configurable": {"thread_id": f"dev_{issue['id']}", "recursion_limit": 25}}
return await compiled.ainvoke(state, config=config)
```

Afterwards `metadata["dev_pipeline"]` holds `iteration`, `review_result.score`, and the generated `code_artifact`.

### Parallel unit testing

After `developer`, control flows to `module_router`, whose `send_edges` entry hands routing to `module_dispatcher`: when the developer produced multiple modules it emits one `Send()` per module to `dev_unit_tester`; otherwise it falls back to the sequential `unit_test_agent`. All parallel workers converge on `dev_test_aggregator`, and both paths feed the same test gate.

### Gates and retry loops

The builder wires the gates as conditional edges. The test gate is a `failure_gate` on the aggregated test report; the review gate is a `score_gate` on the reviewer's verdict:

```python
_TEST_GATE = failure_gate(
    field="dev_pipeline.test_report.failing_tests",
    on_pass="qa_agent",
    on_fail="developer",
    max_iterations=3,
)
_REVIEWER_GATE = score_gate(
    field="dev_pipeline.review_result.score",
    threshold=0.8,
    on_pass="approval_seed",
    on_fail="developer",
    max_iterations=3,
)
```

Both loops return to `developer` (not just the tester), so a failing signal triggers a genuine re-implementation. On review success, `approval_seed` (`seed_hitl_metadata`) writes the `dev_pipeline.code_artifact` field and `risk_level="HIGH"` into metadata, `human_approval_node` raises the interrupt, and `_HITL_GATE` routes approve → END, reject/request_changes → developer — or bypasses the whole flow when `settings.hitl_enabled` is false (CI/CD mode).

## Key API

| Symbol | Role |
|---|---|
| `register_dev_pipeline()` | Idempotent async registration in `SubgraphRegistry` |
| `get_compiled_dev_pipeline()` | Returns the compiled graph, building it on first call |
| `po_agent_node` … `reviewer_agent_node` | The six role agents (PO, Architect, Developer, UnitTest, QA, Reviewer) |
| `module_router_node` / `module_dispatcher` | `Send()` fan-out: one parallel tester per module, or sequential fallback |
| `dev_unit_tester_node` / `dev_test_aggregator_node` | Parallel test workers and their result merger |
| `failure_gate` / `score_gate` | Test gate (failing tests → developer) and review gate (score >= 0.8) |
| `seed_hitl_metadata` / `human_approval_node` / `hitl_gate` | HITL approval flow with `hitl_enabled=False` bypass |
| `metadata["dev_pipeline"]` | `iteration`, `test_report.failing_tests`, `review_result.score`, `code_artifact` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/03_dev_pipeline.ipynb
uv run python examples/subgraphs/03_dev_pipeline.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [ML Pipeline](01_ml_pipeline.md) — the simplest of the three classic pipelines, one score gate.
- [Financial Analyst](02_financial_analyst.md) — data-quality gates plus the same optional HITL flow.
- [Code Review](04_code_review.md) — a standalone linter → security → review → report pipeline.
- [HITL Approval](09_hitl_approval.md) — the approval-gate primitives used here, in isolation.
