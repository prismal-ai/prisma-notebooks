# ML Pipeline

> **Notebook:** [`notebooks/subgraphs/01_ml_pipeline.ipynb`](../../notebooks/subgraphs/01_ml_pipeline.ipynb) · **Example:** [`examples/subgraphs/01_ml_pipeline.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/01_ml_pipeline.py)

The ML Pipeline subgraph runs a complete machine-learning workflow as a chain of six specialist agents: data_ingester → eda_analyst → feature_engineer → model_trainer → model_evaluator → model_exporter. Between evaluation and export sits a quality gate: if the model's primary score falls below 0.7, control loops back to the trainer for another attempt (at most 3 iterations), so a weak model is never silently exported.

Use it when you want an end-to-end train/evaluate/export loop orchestrated by agents — the demo drives it with the classic UCI Heart Disease dataset (303 patients, 14 features, binary classification), a natural benchmark for exercising every stage.

## What it demonstrates

- Registering and compiling a prebuilt subgraph via the `register_ml_pipeline()` + `get_compiled_ml_pipeline()` convenience API.
- A six-node linear pipeline with a `score_gate` retry loop (score >= 0.7, max 3 retrain cycles).
- Passing an ML task description through `state["metadata"]` and an initial `HumanMessage`.
- Reading per-stage results (score, chosen model, iterations, export path) back from `metadata["ml_pipeline"]`.
- Reusing the individual node functions in a custom graph.

## Pipeline topology

```
data_ingester ──► eda_analyst ──► feature_engineer ──► model_trainer ──► model_evaluator
                                                            ▲                  │
                                                            │     [score_gate: primary_score >= 0.7?]
                                                            └──── NO (retrain, max 3) ──┤
                                                                                       YES
                                                                                        ▼
                                                                                 model_exporter ──► END
```

If the score is still below 0.7 after 3 attempts, the gate stops looping and the best model obtained is exported anyway.

## How it works

### Registering and compiling the subgraph

The builder module exposes the two entry points used by all three classic pipelines (01–03): an idempotent async `register_*()` that installs the `SubgraphDefinition` in the `SubgraphRegistry`, and a `get_compiled_*()` that returns the cached `CompiledStateGraph` (building it on first use).

```python
from prismal.agents.state import create_initial_state
from prismal.agents.subgraphs.ml_pipeline.builder import (
    get_compiled_ml_pipeline,
    register_ml_pipeline,
)

await register_ml_pipeline()
subgraph = await get_compiled_ml_pipeline()
```

`register_ml_pipeline()` accepts a `checkpointer_path` (SQLite; use `":memory:"` in tests) and skips work if the pipeline is already registered.

### Describing the ML task

The demo does not ship a CSV — it hands the agents a task descriptor with everything they need to orchestrate the run: dataset metadata, target column, metric, and the algorithms to try.

```python
ML_TASK = {
    "dataset_name": "UCI Heart Disease",
    "dataset_path": "data/heart_disease.csv",  # path the agent would use
    "target_column": "target",
    "task_type": "binary_classification",
    "evaluation_metric": "accuracy",
    "target_score": 0.80,
    "train_test_split": 0.8,
    "algorithms_to_try": ["RandomForest", "GradientBoosting", "LogisticRegression"],
}
```

### Seeding state and invoking

The task goes into `state["metadata"]`, the instruction goes in as a `HumanMessage`, and the run is pinned to a LangGraph thread so checkpointing works:

```python
state = create_initial_state(session_id="nb-ml-pipeline")
state["metadata"] = {
    "task": ML_TASK,
    "pipeline": "ml_pipeline",
    "thread_id": "ml_heart_disease_001",
}
config = {
    "configurable": {
        "thread_id": "ml_heart_disease_001",
        "recursion_limit": 20,
    }
}
final_state = await subgraph.ainvoke(state, config=config)
```

The `recursion_limit` of 20 leaves headroom for the retrain loop (6 nodes plus up to 3 extra trainer/evaluator cycles).

### Reading the results

Each stage writes its artifacts under `metadata["ml_pipeline"]`, so the caller can pull structured metrics rather than parsing prose:

```python
pipeline_meta = final_state.get("metadata", {}).get("ml_pipeline", {})
if pipeline_meta:
    print(f"  Final score   : {pipeline_meta.get('primary_score', 'N/A'):.3f}")
    print(f"  Chosen model  : {pipeline_meta.get('best_model', 'N/A')}")
    print(f"  Iterations    : {pipeline_meta.get('training_iterations', 1)}")
    print(f"  Features used : {pipeline_meta.get('n_features', 'N/A')}")
    print(f"  Exported model: {pipeline_meta.get('export_path', 'N/A')}")
```

### The quality gate

In the builder, the gate is a `score_gate` conditional edge attached to `model_evaluator`. It reads the nested field `ml_pipeline.evaluation_report.primary_score` from state metadata:

```python
_MODEL_QUALITY_GATE = score_gate(
    field="ml_pipeline.evaluation_report.primary_score",
    threshold=0.7,
    on_pass="model_exporter",
    on_fail="model_trainer",
    max_iterations=3,
)
```

`max_iterations=3` is the anti-infinite-loop guard: after three failed retrains the gate routes forward regardless, exporting the best model achieved. The six node functions (`data_ingester_node`, `eda_analyst_node`, `feature_engineer_node`, `model_trainer_node`, `model_evaluator_node`, `model_exporter_node`) are also importable individually from `prismal.agents.subgraphs.ml_pipeline` for use in your own graphs.

## Key API

| Symbol | Role |
|---|---|
| `register_ml_pipeline()` | Idempotent async registration in `SubgraphRegistry` (+ compiled cache) |
| `get_compiled_ml_pipeline()` | Returns the compiled `CompiledStateGraph`, building it on first call |
| `data_ingester_node` … `model_exporter_node` | The six stage node functions, individually reusable |
| `score_gate(...)` | Conditional edge: score >= 0.7 → export, else retrain (max 3) |
| `metadata["ml_pipeline"]` | Per-stage artifacts: `primary_score`, `best_model`, `training_iterations`, `export_path` |
| `metadata["ml_pipeline"]["evaluation_report"]["primary_score"]` | The exact field the quality gate evaluates |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/01_ml_pipeline.ipynb
uv run python examples/subgraphs/01_ml_pipeline.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [Financial Analyst](02_financial_analyst.md) — another multi-gate analysis pipeline, with data-quality and HITL gates.
- [Dev Pipeline](03_dev_pipeline.md) — the software-engineering counterpart, with parallel test fan-out.
- [Data ETL](05_data_etl.md) — extract → validate → transform → load with a validation gate.
- [Analysis Orchestrator](10_analysis_orchestrator.md) — composes analysis subgraphs at a higher level.
