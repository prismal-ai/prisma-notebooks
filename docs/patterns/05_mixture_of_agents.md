# Mixture of Agents (MoA)

> **Notebook:** [`notebooks/patterns/05_mixture_of_agents.ipynb`](../../notebooks/patterns/05_mixture_of_agents.ipynb) · **Example:** [`examples/patterns/05_mixture_of_agents.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/05_mixture_of_agents.py)

Mixture of Agents (SPEC-PAT-006, after Wang et al. 2024) improves answer quality by exploiting model diversity: a pool of *proposer* models each answers a query independently and in parallel, and an *aggregator* model then synthesizes their proposals into one cohesive final answer. Because different models (or the same model sampled independently) have different blind spots, the aggregator can keep the strongest points from each proposal — collective accuracy exceeds any single proposer.

The notebook applies MoA to **MedQA USMLE**, a dataset of 12,723 medical-board multiple-choice questions (pharmacology, pathophysiology, microbiology). Medicine is a natural fit: diverse perspectives reduce clinical reasoning errors, and each question has a verifiable correct option, so the notebook can score whether the synthesized answer lands on it.

## What it demonstrates

- Running N proposer models concurrently and synthesizing their answers with an aggregator via `MixtureOfAgents.generate()`.
- Inspecting `MoAResult`: `final_answer`, per-layer `layer_outputs`, and `providers_used` for the proposers that actually succeeded.
- Fault tolerance semantics: individual proposer failures are dropped silently; `MoAError` is raised only when *every* proposer fails.
- Multi-layer refinement with `n_aggregator_layers > 1`, where each aggregator pass refines the previous synthesis.
- Checking synthesized answers against the dataset's ground-truth options for a simple accuracy metric.

## How it works

### Formatting the medical question

Each MedQA sample carries a clinical vignette, four options, and the correct letter. A small helper renders it as a single prompt asking for step-by-step reasoning:

```python
def format_question(sample: dict) -> str:
    options_text = "\n".join(f"  {k}) {v}" for k, v in sample["options"].items())
    return (
        f"Medical question ({sample['category']}):\n\n"
        f"{sample['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        "Please reason through your answer step by step and choose the correct option."
    )
```

### Configuring proposers and aggregator

`MixtureOfAgents` takes an ordered list of proposer model ids and an aggregator model id (defaulting to `proposer_models[0]` when omitted). The notebook uses three copies of the same model for portability, but the pattern is strongest with genuinely different providers — different training data means different biases:

```python
moa = MixtureOfAgents(
    proposer_models=models,          # e.g. ["gpt-4o", "claude-sonnet-4-6", "gemini-1.5-pro"]
    aggregator_model=aggregator,     # e.g. a higher-capacity model
    n_aggregator_layers=1,
)
result = await moa.generate(
    query=format_question(sample),
    state={"messages": [], "metadata": {"category": sample["category"]}},
)
```

Internally, each proposer is resolved through `ProviderRegistry.get_llm(model)` and invoked in parallel with `asyncio.gather(..., return_exceptions=True)`. Prompts are assembled with `SecurePromptBuilder` — the query is user-controlled content and never f-stringed directly into the system template. The aggregator then receives the query plus the surviving proposals labelled `[Expert 1]`, `[Expert 2]`, … and is instructed to integrate the stronger points and paraphrase rather than copy.

### Reading the result

`MoAResult.layer_outputs[0]` holds the proposer outputs (ordered to match `providers_used`); each subsequent entry is one aggregator pass. The notebook uses this to report success ratios and to verify the correct option:

```python
print(f"Successful proposers: {len(result.layer_outputs[0])}/{len(proposer_models)}")
print(f"Providers used      : {result.providers_used}")

if sample["correct"] in result.final_answer:
    correct_count += 1
```

Because MedQA questions have a known correct letter, the loop across the three samples produces a small accuracy report (`MoA accuracy: 3/3 questions` in a typical run) — a concrete, checkable signal that the synthesis kept the clinically correct option rather than averaging proposals into mush.

### Fault tolerance and multi-layer refinement

Partial failure is a first-class design point, summarized by the notebook as:

```
If 1 of 3 proposers fails → MoA continues with the remaining 2
If 2 of 3 proposers fail  → MoA continues with the remaining 1
If 3 of 3 proposers fail  → MoAError (total failure)
```

Each dropped proposer is logged (`moa_proposer_failed`) and simply omitted from `providers_used`, so the result object always tells you exactly which perspectives fed the synthesis. This matters operationally: mixing providers means mixing failure modes (rate limits, timeouts, region outages), and MoA degrades gracefully instead of failing on the weakest link.

The final section runs a two-layer configuration, where the layer-2 aggregator refines layer-1's synthesis:

```python
moa_multilayer = MixtureOfAgents(
    proposer_models=proposer_models[:2],
    aggregator_model=aggregator_model,
    n_aggregator_layers=2,
)
result_ml = await moa_multilayer.generate(query=format_question(sample), state={})
# result_ml.layer_outputs → [proposals, [layer-1 synthesis], [layer-2 refinement]]
```

`generate()` also accepts an optional `budget_guard_fn`: before each aggregator layer the guard is consulted, and a soft budget cap stops further refinement and returns the best synthesis so far (see the Budget governance article). Each run is traced under the `moa.generate` OpenTelemetry span, with proposer count, layer count, and answer length as attributes.

A practical rule of thumb: extra aggregator layers give diminishing returns compared to extra *diverse* proposers. Reach for `n_aggregator_layers=2` only when the first synthesis tends to be verbose or internally inconsistent; reach for a different proposer model whenever you can.

## Key API

| Symbol | Role |
|---|---|
| `MixtureOfAgents` | The pattern class: parallel proposers + N aggregator passes; validates `proposer_models` non-empty and `n_aggregator_layers >= 1`. |
| `MixtureOfAgents.generate()` | `(query, state, budget_guard_fn=None) -> MoAResult`; raises `MoAError` only on total proposer failure. |
| `MoAResult` | `final_answer`, `layer_outputs` (index 0 = proposals, then one entry per aggregator pass), `providers_used` (successful proposer ids). |
| `MoAError` | Raised when every proposer fails to produce an output. |
| `ProviderRegistry` | Resolves each model id to an LLM client — proposers can span different providers. |
| `SecurePromptBuilder` | Isolates the user query and proposal text inside the proposer/aggregator prompts. |

## Run it

```bash
uv run jupyter lab notebooks/patterns/05_mixture_of_agents.ipynb
# or the script version:
uv run python examples/patterns/05_mixture_of_agents.py   # from the prismal repo
```

Requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — every proposer and aggregator call is a real LLM invocation.

## Related

- [Debate](02_debate.md) — agents that argue *with* each other over multiple rounds, versus MoA's independent proposals synthesized once.
- [Swarm](08_swarm.md) — decentralized agent handoff; another way to combine multiple agents, but sequential rather than parallel.
- [Kokoro deliberation](../kokoro_deliberation.md) — persona-driven souls arguing toward agreement with a single accountable judge, a deliberative cousin of MoA's aggregator.
- [Budget governance](../budget_governance.md) — how `budget_guard_fn` caps MoA's aggregator layers under a token/cost budget.
