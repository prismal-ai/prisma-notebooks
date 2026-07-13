# MetaLearner: self-scoring traces and the human-gated improvement cycle

> **Notebook:** [`notebooks/meta_learner.ipynb`](../notebooks/meta_learner.ipynb) · **Example:** [`examples/meta_learner.py`](https://github.com/prismal-ai/prismal/blob/main/examples/meta_learner.py)

`MetaLearner` (`prismal/agents/meta_learner.py`, tracked as T-212) is prismal's self-improvement
loop: it fetches recent execution traces from Langfuse, has the LLM self-score each one on task
completion, accuracy, and efficiency, and — for traces that fall below a threshold — generates
concrete improvement proposals (system prompt edits, new skills, supervisor routing changes). It
is one of the 26 specialist agents behind the supervisor, and its defining property is that it
**never applies changes itself**: proposals land on disk next to a `human_review_required.txt`
sentinel, and a human decides what ships.

This notebook runs the full cycle offline. A subclass replaces the Langfuse trace source with a
canned list, a fake provider registry replaces the evaluator LLM, and proposals are written to a
temporary directory — never into `skills/custom/`. That means you can study score → flag →
propose → gate end to end with zero credentials.

## What it demonstrates

- The full review cycle: `fetch_traces()` → `score_traces()` → `generate_proposals()` →
  `review()` with the human-review sentinel written last.
- `TraceScore`, the Pydantic self-assessment model (three 0.0–1.0 dimensions, a weighted
  `overall`, and an `issues` list), and how the `score_threshold` flags low performers.
- The two injection seams the demo uses: overriding `fetch_traces()` in a subclass, and swapping
  the module-level `ProviderRegistry` for a deterministic fake.
- The on-disk artefacts behind the gate: timestamped `proposals_*.md` reports plus the
  `human_review_required.txt` sentinel.

## How it works

### Canned traces and deterministic scores

The demo defines three fake traces — two good runs and one failure (`trace-002`, a unit
conversion the agent could not do) — and a matching table of the self-assessments the evaluator
LLM would emit:

```python
FAKE_SCORES: dict[str, dict[str, Any]] = {
    "trace-001": {"task_completion": 0.95, "accuracy": 0.9, "efficiency": 0.85, "issues": []},
    "trace-002": {
        "task_completion": 0.2,
        "accuracy": 0.4,
        "efficiency": 0.3,
        "issues": ["failed to answer", "missing unit-conversion skill"],
    },
    "trace-003": {"task_completion": 0.9, "accuracy": 0.95, "efficiency": 0.8, "issues": []},
}
```

In production, `fetch_traces(days=7)` queries the Langfuse HTTP API using the configured
credentials, and returns an empty list (never raising) when Langfuse is disabled or misconfigured.

### The offline fakes

The fake LLM answers both roles the real one plays. `score_traces()` sends one prompt per trace
containing a `TRACE ID:` marker, while `generate_proposals()` sends a single summary prompt — so
the fake dispatches on that marker:

```python
class _FakeLLM:
    """Deterministic stand-in for the evaluator/proposal LLM."""

    async def ainvoke(self, messages: list[Any]) -> Any:
        prompt = str(messages[0].content)
        match = re.search(r"TRACE ID: (\S+)", prompt)
        # A "TRACE ID:" prompt is score_traces(); otherwise generate_proposals().
        content = json.dumps(FAKE_SCORES.get(match.group(1), {})) if match else FAKE_PROPOSALS
        ...
```

The trace source is replaced the cleanest way possible — a subclass overriding one method:

```python
class OfflineMetaLearner(MetaLearner):
    """MetaLearner whose trace source is a canned list instead of Langfuse."""

    async def fetch_traces(self, days: int = 7) -> list[dict[str, Any]]:
        """Return the fake traces (production queries the Langfuse API)."""
        return FAKE_TRACES
```

Because `MetaLearner` wires `ProviderRegistry().get_llm()` internally (the same lazy-default
pattern the advanced-architecture modules use), the demo swaps the module attribute for a
`FakeProviderRegistry`. A real deployment leaves the registry untouched and the configured
provider is used.

### Scoring and flagging

`score_traces()` truncates each trace's input/output to 500 characters, asks the LLM for a JSON
verdict, and computes `overall` as the mean of the three dimensions. Anything below
`score_threshold` (default 0.6) is a low performer. The notebook prints each score with a flag:

```python
learner = OfflineMetaLearner(proposals_dir=proposals_dir, score_threshold=0.6)

traces = await learner.fetch_traces(days=7)
scores = await learner.score_traces(traces)
for score in scores:
    _print_score(score)   # [LOW ] trace-002: overall=0.30 ...
```

A scoring failure on one trace is logged and skipped rather than aborting the batch — the cycle
degrades gracefully, mirroring `fetch_traces()`.

### The review cycle and the human gate

`review()` orchestrates everything. If no trace is low-scoring it returns a clean bill of health;
otherwise it asks the LLM for a Markdown report covering root causes, exact system prompt edits,
new skills to add, and supervisor routing improvements, then writes the artefacts:

```python
summary = await learner.review(days=7)
print(summary)

proposal_files = sorted(proposals_dir.glob("proposals_*.md"))
sentinel = proposals_dir / "human_review_required.txt"
```

The sentinel is the gate: proposals are advisory text on disk, the summary ends with
`ACTION REQUIRED`, and nothing is applied to prompts, routing, or skills until a human reviews
the `proposals_*.md` files and deletes the sentinel. This is the same review-before-activation
discipline the skill tiers enforce — AI-generated output goes to a gitignored staging area
(`skills/custom/meta_proposals/` by default; a temp directory here), never straight into the
committed `available/` tier.

## Key API

| Symbol | Role |
|---|---|
| `MetaLearner(proposals_dir=None, score_threshold=0.6)` | Orchestrates the review cycle; defaults to writing under `skills/custom/meta_proposals/`. |
| `MetaLearner.fetch_traces(days=7)` | Pulls recent traces from Langfuse; empty list when disabled or on error (never raises). |
| `MetaLearner.score_traces(traces)` | LLM self-scores each trace; returns `TraceScore` objects. |
| `MetaLearner.generate_proposals(low_scores)` | Asks the LLM for a Markdown improvement report for flagged traces. |
| `MetaLearner.review(days=7)` | Full cycle: fetch → score → propose → save with the human-review sentinel; returns a summary string. |
| `TraceScore` | Pydantic model: `task_completion` / `accuracy` / `efficiency` / `overall` (all 0.0–1.0) plus `issues`. |

## Run it

```bash
uv run jupyter lab notebooks/meta_learner.ipynb
uv run python examples/meta_learner.py   # from the prismal repo
```

No API key required — the demo runs fully offline with fake traces and a fake LLM registry.

## Related

- [Skill creator agent](skill_creator_agent.md) — the agent that turns "new skills to add" proposals into actual skill code, behind the same human-review gate.
- [Agent eval](agent_eval.md) — systematic evaluation of agent behavior, the offline counterpart to trace self-scoring.
- [Observability integration](observability_integration.md) — the Langfuse/OTel layer that produces the traces MetaLearner consumes.
- [Supervisor quickstart](supervisor_quickstart.md) — the routing graph whose prompts and routes the proposals target.
