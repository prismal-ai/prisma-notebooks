# Data ETL + EDA

> **Notebook:** [`notebooks/subgraphs/05_data_etl.ipynb`](../../notebooks/subgraphs/05_data_etl.ipynb) · **Example:** [`examples/subgraphs/05_data_etl.py`](https://github.com/prismal-ai/prismal/blob/main/examples/subgraphs/05_data_etl.py)

The `data_etl` subgraph is a five-node data pipeline with a built-in quality gate: an `extractor` loads data into a Polars DataFrame, a `validator` runs schema and sanity checks, and a conditional edge then either continues to `transformer → loader → auditor` (validation passed) or jumps straight to `auditor` (validation failed) — so the transform and load stages are skipped and the audit records the failure. Every stage writes into `state["metadata"]["data_etl"]`.

This walkthrough runs the pipeline on the classic Titanic dataset, using the ETL stages as an Exploratory Data Analysis (EDA) exercise: null analysis in the validator, median imputation and feature engineering in the transformer, and a before/after quality report in the auditor. Use this subgraph when you need auditable data pipelines where bad input must never reach the transform stage.

## What it demonstrates

- The validation gate: `make_etl_branching_gate()` routes `validator → transformer` on pass and `validator → auditor` on fail — demonstrated with both a clean run and a deliberately broken dataset.
- Factory injection of every stage (`extractor_fn`, `validator_fn`, `transformer_fn`, `loader_fn`) so the whole pipeline runs on in-memory data with no external I/O.
- Composing a custom transformer on top of the built-in declarative transforms (`select` / `filter` / `rename`) with EDA steps: imputation, encoding, derived features.
- Building via `build_data_etl_subgraph()` → `SubgraphDefinition` → `assemble_state_graph(defn).compile()`.
- Reading pipeline outputs (`dataframe`, `validation`, `transform_log`, `loaded_row_count`) from `state["metadata"]["data_etl"]`.

## Pipeline topology

```
extractor → validator ──(passed)──► transformer → loader → auditor
                 │
                 └──(failed)──────────────────────────────► auditor
```

## How it works

### Dataset and pipeline spec

The Titanic subset is embedded as inline CSV (50 passengers, with real nulls in `Age`). The declarative part of the pipeline is a list of transform specs the default transformer understands:

```python
REQUIRED_COLUMNS = ["PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare"]

TRANSFORMS = [
    {"op": "select", "columns": ["PassengerId", "Survived", "Pclass", "Sex",
                                 "Age", "SibSp", "Parch", "Fare", "Embarked"]},
    {"op": "filter", "column": "Fare", "operator": ">", "value": 0.0},
    {"op": "rename", "mapping": {"PassengerId": "passenger_id", "Survived": "survived",
                                 # ... remaining columns to snake_case
                                 }},
]
```

### Injectable callables

The extractor and loader are trivial in-memory stand-ins (in production they would read/write CSV, Parquet or a database). The validator goes beyond the default shape check and performs EDA-style checks — nulls per column, logical ranges, expected cardinalities:

```python
def titanic_validator(df: pl.DataFrame, source: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validator with EDA analysis: schema, nulls, ranges."""
    errors: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if df.height == 0:
        errors.append("Dataset is empty")
        return False, errors

    for col in df.columns:
        null_pct = df[col].null_count() / df.height * 100
        if null_pct > 80:
            errors.append(f"Column '{col}' has {null_pct:.1f}% nulls (> 80%)")
    # ... range checks on Age/Fare, cardinality checks on Survived/Pclass
    return (not errors, errors)
```

The custom transformer first delegates to the built-in `_default_transformer` for the declarative ops, then layers EDA feature engineering on top — age imputed by per-class median, binary sex encoding, `family_size`, `age_group` buckets, `log1p(fare)`:

```python
async def titanic_transformer(
    df: pl.DataFrame, transforms: list[dict[str, Any]]
) -> tuple[pl.DataFrame, list[str]]:
    from prismal.agents.subgraphs.data_etl.transformer_node import _default_transformer

    current, log = _default_transformer(df, transforms)

    # Impute age with the median of each passenger class
    medians = current.group_by("pclass").agg(pl.col("age").median().alias("age_median"))
    # ... build median_map, fill nulls via pl.when(pl.col("age").is_null())

    # Feature engineering
    current = current.with_columns((pl.col("sibsp") + pl.col("parch") + 1).alias("family_size"))
    # ... sex_bin encoding, age_group buckets, fare_log1p, embarked mode imputation
    return current, log
```

Each step appends to the `transform_log`, which the auditor later surfaces — the pipeline is self-documenting.

### Build, compile, invoke

The builder wires the five nodes plus the conditional edge on `validator`; `required_columns` parametrizes the default validator if you don't inject your own. As with all subgraphs in this family, `build_data_etl_subgraph()` returns a `SubgraphDefinition` and `assemble_state_graph(...)` compiles it:

```python
await register_data_etl()
subgraph = build_data_etl_subgraph(
    extractor_fn=titanic_extractor,
    validator_fn=titanic_validator,
    transformer_fn=titanic_transformer,
    loader_fn=memory_loader,
    required_columns=REQUIRED_COLUMNS,
)
graph = assemble_state_graph(subgraph).compile()

state = create_initial_state(session_id="nb-data-etl")
state["metadata"] = {
    "data_etl": {
        "source": {"type": "inline", "name": "titanic"},
        "transforms": TRANSFORMS,
        "destination": {"type": "memory"},
    }
}
final_state = await graph.ainvoke(state, config={"configurable": {"thread_id": "etl_titanic_001"}})
```

The `source`, `transforms` and `destination` specs travel in metadata; the extractor, transformer and loader each read their spec from there.

### The validation gate, exercised both ways

In the graph, the gate is a conditional-edge function returned by `make_etl_branching_gate()` — it reads `metadata.data_etl.validation` and returns the next node name (`"transformer"` on pass, `"auditor"` otherwise; a missing validation record also routes to the auditor). The notebook demonstrates the error path with a dataset that lacks every required column:

```python
bad_csv = "col_a,col_b\n1,foo\n2,bar\n"
df_bad = pl.read_csv(io.StringIO(bad_csv))
passed, errors = titanic_validator(df_bad, source)
# → FAIL: "Missing required columns: [...]"
# → Gate redirects to auditor (error path) — transformer is skipped
```

After a successful run, the auditor compares before/after shape and null counts, and the notebook finishes with survival-rate EDA over the engineered features (by class, sex, age group and family size).

## Key API

| Symbol | Role |
|---|---|
| `build_data_etl_subgraph(...)` | Builds the `SubgraphDefinition` (5 nodes + conditional edge on `validator`) |
| `register_data_etl()` | Idempotent registration in `SubgraphRegistry` (default Polars I/O) |
| `assemble_state_graph(defn)` | Compiles the definition into a runnable `StateGraph` |
| `extractor_fn` | `async (source) -> pl.DataFrame` — data ingestion |
| `validator_fn` | `(df, source) -> (bool, list[str])` — schema/quality checks (e.g. Pandera) |
| `make_etl_branching_gate()` | Conditional edge: pass → `transformer`, fail → `auditor` |
| `transformer_fn` | `async (df, transforms) -> (df, log)` — cleaning + feature engineering |
| `loader_fn` | `async (df, destination) -> int` — persistence, returns rows written |
| `state["metadata"]["data_etl"]` | Carries `source`, `transforms`, `destination`, `dataframe`, `validation`, `transform_log`, `loaded_row_count` |

## Run it

```bash
uv run jupyter lab notebooks/subgraphs/05_data_etl.ipynb
uv run python examples/subgraphs/05_data_etl.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [ML Pipeline](01_ml_pipeline.md) — the natural downstream consumer: Ingester → EDA → Features → Trainer.
- [Code Review](04_code_review.md) — a linear sibling pipeline (no gate) with the same factory-injection pattern.
- [Customer Service](06_customer_service.md) — another gated subgraph, routing on confidence instead of validation.
