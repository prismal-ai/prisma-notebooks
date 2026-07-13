# MultimodalFusion — Combining Observations Across Modalities

> **Notebook:** [`notebooks/multimodal/05_multimodal_fusion.ipynb`](../../notebooks/multimodal/05_multimodal_fusion.ipynb) · **Example:** [`examples/multimodal/05_multimodal_fusion.py`](https://github.com/prismal-ai/prismal/blob/main/examples/multimodal/05_multimodal_fusion.py)

`MultimodalFusion` (SPEC-MM-AGT-005) is the last stage of prismal's opt-in Fase F multimodal layer (gated by `settings.multimodal_enabled`, default `False`): once the vision, audio, video, and text agents have each produced their view of a scene, fusion combines those `ModalContribution`s into one answer. It offers three strategies with an explicit cost ladder — `concat` (deterministic, zero LLM calls), `moderator` (one LLM synthesis call), and `moa` (delegates to the `MixtureOfAgents` pattern: N proposers plus an aggregator).

The demo runs all three strategies over VQA-style scenes without touching a model: `moderator` receives an injected `moderator_fn`, and `moa` receives a duck-typed stand-in for `MixtureOfAgents`. No API keys, no provider SDKs — everything is offline. That is the same factory-injection principle used across Fase F: the fusion logic (prompt construction, strategy dispatch, result shaping) is fully exercised while the model boundary is a fake; the layered defaults wire `get_multimodal_llm()` and a settings-derived `MixtureOfAgents` lazily when nothing is injected.

## What it demonstrates

- Three VQA-inspired scenes, each with 2–3 `ModalContribution`s (vision, audio, video, text) carrying `content`, `agent_id`, and `confidence`.
- The `concat` strategy: labels each contribution `[modality · agent_id · conf=…]` and concatenates — ideal for tests and as a no-cost fallback.
- The `moderator` strategy through an injected `moderator_fn` that parses the synthesis prompt deterministically instead of calling an LLM.
- The `moa` strategy through a duck-typed fake exposing `await moa.generate(prompt, state)` → object with `.final_answer` — the only surface `_combine_moa` actually needs.
- Constructor validation: an unknown strategy raises `ValueError` immediately, not at combine time.

## How it works

### Contributions as first-class data

Each scene models a Visual Question Answering setup: a question plus what each modal agent observed, with a confidence the fusion output preserves in its labels:

```python
ModalContribution(
    modality=Modality.IMAGE,
    content="A golden retriever running across wet sand near the shoreline.",
    agent_id="vision",
    confidence=0.92,
),
ModalContribution(
    modality=Modality.AUDIO,
    content="Sound of waves crashing and faint barking.",
    agent_id="audio",
    confidence=0.78,
),
```

All strategies share one rendering (`_render_contributions`) and one prompt builder (`_build_prompt`), so switching strategies changes *who synthesizes*, never *what they see*. The optional `context=` argument (here, the VQA question) is appended to the synthesis prompt.

### The three strategies and their cost ladder

```python
for strategy in ("concat", "moderator", "moa"):
    kwargs = {}
    if strategy == "moderator":
        kwargs["moderator_fn"] = fake_moderator
    if strategy == "moa":
        kwargs["moa"] = make_fake_moa()
    fusion = MultimodalFusion(strategy=strategy, **kwargs)
    result = await fusion.combine(scene["contributions"], context=scene["question"])
    print(result.strategy_used, result.answer)
```

- **`concat`** performs no calls at all: the answer *is* the labelled rendering. Deterministic, free, and the right degradation target when budgets or backends are unavailable.
- **`moderator`** builds the synthesis prompt and makes exactly one call. With no `moderator_fn` injected, the default routes the prompt through `SecurePromptBuilder` to `get_multimodal_llm()` — contributions contain captions and transcripts, which are user-controlled content and must not be interpolated into raw templates.
- **`moa`** hands the same prompt to `MixtureOfAgents` (from `agents/patterns/mixture_of_agents.py`): parallel proposers each draft an answer, an aggregator synthesizes them. Highest quality, N+1 calls.

### Faking the MoA without LLMs

The real `MixtureOfAgents` needs model ids configured in the provider registry, but `_combine_moa` only requires two things — an awaitable `generate(prompt, state)` and a `.final_answer` attribute on the result. A tiny stand-in satisfies the contract:

```python
class FakeMoA:
    """Duck-typed stand-in for MixtureOfAgents for the `moa` fusion."""

    def __init__(self, n_proposers: int = 3) -> None:
        self._n = n_proposers

    async def generate(self, query: str, state) -> _FakeMoAResult:
        proposals = [f"[Expert {i + 1}] view of: {query[:60]}" for i in range(self._n)]
        synth = (
            f"MoA-synth ({self._n} proposers): integrated answer drawing from "
            + "; ".join(p.split(": ", 1)[1] for p in proposals)
        )
        return _FakeMoAResult(final_answer=synth)
```

This is duck typing as a testing seam: because the fusion class depends on a behavioral contract rather than a concrete class, the demo (and the test suite) swap the expensive pattern for a three-line fake. When no `moa` is injected, `_default_moa()` builds one lazily with `proposer_models=[settings.multimodal_model]`.

### Failure modes and validation

Strategy validation is eager — `MultimodalFusion(strategy="ensemble")` raises `ValueError` at construction, so a typo cannot survive until runtime. At combine time, a failing moderator or MoA call is wrapped in `MultimodalFusionError` with the original exception chained; `concat` cannot fail. Every `combine()` runs inside an OTel span (`mm.fusion.combine`) tagged with the strategy, and the returned `FusionResult` echoes `strategy_used` so callers can log which rung of the cost ladder actually answered.

### Choosing a strategy

The three strategies form a deliberate quality/cost ladder, and the right rung depends on where fusion sits in your pipeline:

- Use **`concat`** in tests, in CI, and as the degradation target when a budget guard trips or no backend is configured — it is pure data transformation.
- Use **`moderator`** (the default) for production single-answer fusion: one call, coherent prose, and the injectable `moderator_fn` keeps it swappable.
- Reserve **`moa`** for high-stakes queries where multiple independent drafts measurably improve the synthesis — it multiplies the call count by the number of proposers plus one.

Because the strategy is a constructor argument and all three consume identical contributions, upgrading or downgrading is a one-line change with no impact on the surrounding pipeline.

Note that fusion is downstream of the security gates: by the time contributions exist, `MediaValidator.validate()` already screened the raw media (before any model call), and in graph deployments the media itself traveled as path-based descriptors under `state["metadata"]["mm"]` — never as raw bytes in checkpointed state.

## Key API

| Symbol | Role |
|---|---|
| `MultimodalFusion` | Fusion engine; constructor takes `strategy`, `moa`, `moderator_fn`, `settings` (keyword-only); default strategy is `"moderator"`. |
| `MultimodalFusion.combine(contributions, *, context=None)` | Fuses contributions; returns `FusionResult`. |
| `ModalContribution` | Frozen dataclass: `modality`, `content`, `agent_id`, `confidence`. |
| `FusionResult` | Frozen dataclass: `answer`, `contributions`, `strategy_used`. |
| `strategy="concat"` | Deterministic labelled concatenation; zero LLM calls. |
| `strategy="moderator"` | One synthesis call via `moderator_fn` (`async (prompt) -> str`); default wires `get_multimodal_llm` + `SecurePromptBuilder`. |
| `strategy="moa"` | Delegates to `MixtureOfAgents.generate(prompt, state)`; N proposers + aggregator; built lazily from `settings.multimodal_model` if not injected. |
| `Modality` | Contribution modality enum, shared with the modality router. |
| `MultimodalFusionError` | Wraps moderator/MoA failures; unknown strategies raise `ValueError` at construction. |

## Run it

```bash
uv run jupyter lab notebooks/multimodal/05_multimodal_fusion.ipynb
uv run python examples/multimodal/05_multimodal_fusion.py   # from the prismal repo
```

No API keys required — the notebook injects a fake moderator function and a duck-typed MoA stand-in, running fully offline.

## Related

- [ModalityRouter](04_modality_router.md)
- [VideoAgent](03_video_agent.md)
- [Multimodal pipeline subgraph](../multimodal_pipeline.md)
- [Multimodal RAG](../rag/09_multimodal_rag.md)
