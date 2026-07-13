# Code Review

> **Notebook:** [`notebooks/subgraphs/04_code_review.ipynb`](../../notebooks/subgraphs/04_code_review.ipynb) · **Example:** [`examples/subgraphs/04_code_review.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/04_code_review.py)

The `code_review` subgraph is a five-stage automated Python code review pipeline. Three analyzer nodes run in sequence — `linter` (style, docstrings, magic numbers), `security_scanner` (injection, hardcoded credentials, unsafe deserialization) and `logic_reviewer` (division by zero, unchecked indices, unreachable branches) — each appending its findings to a shared `issues` list in state metadata. A `suggester` then drafts a remediation per issue, and a terminal `report_generator` computes a severity-weighted score and an approved/rejected verdict against a configurable threshold.

Use it when you want a reproducible, traceable review gate in CI or agent workflows: each analyzer is a factory-injected async callable, so you can plug in flake8, Bandit, or an LLM reviewer per stage — or run entirely offline with heuristics, as this demo does. The dataset is a set of CodeSearchNet-style Python snippets seeded with real-world bug patterns.

## What it demonstrates

- Building a linear five-node review pipeline with `build_code_review_subgraph()` and compiling the returned `SubgraphDefinition` with `assemble_state_graph(defn).compile()`.
- Factory injection of all four analysis stages (`linter_fn`, `scanner_fn`, `reviewer_fn`, `suggester_fn`) so the pipeline runs without any LLM backend.
- The severity-weighted scoring model of `report_generator`: `score = 1.0 - Σ weight[severity]`, with `approved = score >= approval_threshold`.
- Regex-based heuristic detectors for SQL injection, hardcoded credentials, `pickle.loads`, `yaml.load` without a Loader, and `shell=True`.
- Reading results back from `state["metadata"]["code_review"]` (`issues`, `suggestions`, `report`).

## Pipeline topology

```
linter → security_scanner → logic_reviewer → suggester → report_generator
```

A strictly linear pipeline (entry point: `linter`, no conditional edges). Each analyzer accumulates into the shared `issues` list; only `report_generator` renders a verdict.

## How it works

### Imports and severity weights

The notebook imports the builder and the shared `assemble_state_graph` factory, plus `CodeIssue`, the value object every analyzer emits. The weights mirror the ones hard-coded in `report_generator_node.py` — a single critical issue (0.40) alone fails the default 0.8 approval gate:

```python
from prismal.agents.subgraphs.code_review.types import CodeIssue
from prismal.agents.subgraphs.code_review.builder import (
    build_code_review_subgraph,
    register_code_review,
)
from prismal.agents.subgraphs.factory import assemble_state_graph

SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 0.40,
    "high": 0.20,
    "medium": 0.10,
    "low": 0.05,
    "info": 0.01,
}
```

### Dataset: snippets with intentional issues

Five snippets cover the detection surface of each node: `db_utils.py` (SQL injection, hardcoded password, `pickle.loads`), `math_ops.py` (ZeroDivision, IndexError, duplicated `elif`), `file_handler.py` (shell injection, path traversal), `auth_service.py` (clean code — the control case that should be approved), and `data_pipeline.py` (mixed severities including a missing-tests finding).

### Injectable callables

Each stage is an async callable with the signature the builder expects. The heuristic security scanner, for example, is a table of regex patterns mapped to severity, category and a remediation:

```python
async def heuristic_scanner(code: str, filename: str) -> list[CodeIssue]:
    """Heuristic security scanner: detects vulnerability patterns."""
    issues: list[CodeIssue] = []
    lines = code.splitlines()

    patterns = [
        (
            r"""(execute|cursor\.execute)\s*\(\s*["'][^"']*["']\s*\+""",
            "critical", "security",
            "SQL injection: direct interpolation of variables in query.",
            "Use parameterized queries: cursor.execute(sql, (param,))",
        ),
        (
            r"""(?i)(password|secret|api_key|token|credential)\s*=\s*["'][^"']{6,}["']""",
            "critical", "security",
            "Hardcoded credential in the source code.",
            "Use environment variables: os.environ.get('SECRET_KEY') or python-dotenv.",
        ),
        # ... pickle.loads, yaml.load without Loader, shell=True, requests without timeout
    ]
```

`heuristic_linter` flags missing docstrings, magic numbers and over-long lines; `heuristic_reviewer` flags logic errors plus a module-level "no tests" finding; `heuristic_suggester` formats one remediation string per issue.

### Build, compile, invoke

The notebook builds the definition with the callables injected, compiles it through the shared factory, and seeds the code under the `code_review` metadata key. Passing `linter_fn=None` (or any stage as `None`) makes that node fall back to the default LLM from `ProviderRegistry` — which is why an API key is needed for the real run:

```python
await register_code_review(approval_threshold=approval_threshold)
subgraph_def = build_code_review_subgraph(
    linter_fn=heuristic_linter,
    scanner_fn=heuristic_scanner,
    reviewer_fn=heuristic_reviewer,
    suggester_fn=heuristic_suggester,
    approval_threshold=approval_threshold,
)
graph = assemble_state_graph(subgraph_def).compile()

state = create_initial_state(session_id="nb-code-review")
state["messages"] = [HumanMessage(content=f"Review the code of {filename}:\n\n{code}")]
state["metadata"] = {"code_review": {"code": code, "filename": filename, "issues": []}}

config = {"configurable": {"thread_id": f"cr_{snippet['id']}_001"}}
final_state = await graph.ainvoke(state, config=config)
```

Note the pattern: `build_code_review_subgraph()` returns a `SubgraphDefinition` (nodes, edges, entry point) — not a compiled graph — and `assemble_state_graph(defn).compile()` turns it into a runnable LangGraph.

### The verdict

`report_generator` aggregates all accumulated issues into a `CodeReviewReport` and appends an `AIMessage` summary. The scoring is deterministic:

```python
def compute_score(issues: list[CodeIssue]) -> float:
    """Calculate the review score (1.0 = no issues)."""
    penalty = sum(SEVERITY_WEIGHTS.get(issue.severity, 0.0) for issue in issues)
    return max(0.0, 1.0 - penalty)
```

With the default threshold of 0.8: one critical (score 0.60) or two high (0.60) issues reject the code; four medium (0.60) also fail, while up to four low issues (0.80) still pass. The demo prints per-file verdicts, severity/category distributions across all snippets, and a comparison against manual review and single-purpose SAST tools.

## Key API

| Symbol | Role |
|---|---|
| `build_code_review_subgraph(...)` | Builds the `SubgraphDefinition` with injected callables and `approval_threshold` |
| `register_code_review(approval_threshold=0.8)` | Idempotent registration in `SubgraphRegistry` |
| `assemble_state_graph(defn)` | Shared factory turning a `SubgraphDefinition` into a `StateGraph` |
| `linter_fn` / `scanner_fn` / `reviewer_fn` | `async (code, filename) -> list[CodeIssue]` — one per analyzer node |
| `suggester_fn` | `async (code, issues) -> list[str]` — remediation per issue |
| `make_report_generator_node(approval_threshold)` | Terminal node: severity-weighted score + `CodeReviewReport` |
| `CodeIssue` / `CodeReviewReport` | Value objects for findings and the final verdict |
| `state["metadata"]["code_review"]` | Carries `code`, `filename`, `issues`, `suggestions`, `report` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/04_code_review.ipynb
uv run python examples/subgraphs/04_code_review.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [Dev Pipeline](03_dev_pipeline.md) — the full PO → Architect → Developer → QA → Reviewer development pipeline this review stage complements.
- [Data ETL + EDA](05_data_etl.md) — the sibling pipeline with a conditional validation gate instead of a linear flow.
- [Engineering Orchestrator](11_engineering_orchestrator.md) — composes engineering subgraphs like this one at a higher level.
