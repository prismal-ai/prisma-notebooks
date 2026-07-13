# Multimodal Pipeline Subgraph End-to-End

> **Notebook:** [`notebooks/multimodal_pipeline.ipynb`](../notebooks/multimodal_pipeline.ipynb) · **Example:** [`examples/multimodal_pipeline.py`](https://github.com/prismal-ai/prismal/blob/main/examples/multimodal_pipeline.py)

This walkthrough runs an image through prismal's `multimodal_pipeline` subgraph — the Fase F composition `router → [vision | audio | video | text] → fusion → output_formatter` — step by step, so you can watch each node's state update instead of only seeing the final answer. The multimodal layer is **opt-in by design**: it is gated by `settings.multimodal_enabled` (default `False`), and with the flag off the 26 text agents behave exactly as before. Because every agent and the fusion strategy are injectable, this demo swaps the real `VisionAgent` for an `AsyncMock` that returns a fixed caption — no VLM, no Whisper, no FFmpeg, no network, no API keys. Everything runs offline.

Use this pattern whenever you want to understand (or test) the pipeline's topology and state contract before wiring real providers: the same `SubgraphDefinition` that the supervisor mounts when `multimodal_enabled=True` is built here directly, just with fakes injected.

## What it demonstrates

- Building the multimodal pipeline with `build_multimodal_subgraph()` and injecting a fake `VisionAgent` plus the deterministic `concat` fusion strategy (no LLM calls).
- How the `router` node classifies the incoming modality and how the conditional edge dispatches to the matching modal node.
- The multimodal state contract: media in, contributions accumulated, fused answer out — all namespaced under `state["metadata"]["mm"]`.
- Driving the `SubgraphDefinition` nodes manually, exactly the way the compiled subgraph would sequence them.
- Reading the final `AIMessage` produced by `output_formatter_node`.

## How it works

### Fake providers and inline media

The demo needs only two things: a byte string that *looks* like a PNG, and a stand-in vision agent. Eight magic bytes are enough for `MediaValidator`-style sniffing and for `classify_modality()` to recognize an image; the `AsyncMock` mimics `VisionAgent.analyze()` by returning a fixed `VisionResult`.

```python
from prismal.agents.multimodal import VisionResult
from prismal.agents.subgraphs.multimodal_pipeline import build_multimodal_subgraph

# 1×1 transparent PNG-ish header (enough for MediaValidator magic-byte sniffing).
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _fake_vision_agent() -> AsyncMock:
    """A VisionAgent stand-in that returns a fixed caption (no VLM call)."""
    agent = AsyncMock()
    agent.analyze = AsyncMock(
        return_value=VisionResult(
            description="A golden retriever running on a sandy beach.",
            objects=[],
            ocr_text=None,
            model_used="mock-vlm",
        )
    )
    return agent
```

This is the standard Fase F testing seam: `build_multimodal_subgraph()` accepts `vision_agent`, `audio_agent`, `video_agent`, and `fusion` as keyword arguments and only constructs the real agents lazily when you pass `None`. Injecting fakes keeps provider SDKs out of the picture entirely.

### Building the subgraph

```python
definition = build_multimodal_subgraph(
    vision_agent=_fake_vision_agent(),
    fusion_strategy="concat",  # deterministic, no LLM
)
```

The returned `SubgraphDefinition` carries the full topology: `entry_point="router"`, seven nodes (`router`, `vision_node`, `audio_node`, `video_node`, `text_node`, `fusion_node`, `output_formatter_node`), static edges from every modal node into `fusion_node` and on to `output_formatter_node`, and one conditional edge out of `router`. The `concat` fusion strategy joins contributions deterministically, so the run is fully reproducible; `moderator` and `moa` are the LLM-backed alternatives.

### Supplying input state

The caller places media under `metadata.mm.media` and attaches a MIME hint to the message. All multimodal state lives under the `mm` key precisely to isolate the Fase F layer from the rest of `AgentState`:

```python
state: dict = {
    "messages": [
        HumanMessage(
            content="What is in this image?",
            additional_kwargs={"attachments": [{"mime_type": "image/png"}]},
        )
    ],
    "metadata": {"mm": {"media": PNG, "preferred_output": "text"}},
}
```

Two forms of `metadata.mm.media` are supported. This demo uses the direct form (raw bytes) for brevity; in production you should use `ingest_media()` instead, which validates the media, strips EXIF, spills the bytes to a content-addressed file, and records a **path-based descriptor list** — so raw bytes never end up in LangGraph's checkpointed state (see the [ingestion article](multimodal_ingestion.md)). The pipeline's internal `_media_of()` helper resolves both forms transparently.

### Walking the nodes: router → vision → fusion → output

Rather than compiling the graph, the demo drives the nodes in the same order the compiled subgraph would, which makes each intermediate state visible:

```python
state.update(await definition.nodes["router"](state))
next_node = definition.conditional_edges["router"](state)
print("router → ", next_node)

state["metadata"].update((await definition.nodes[next_node](state))["metadata"])
state["metadata"].update((await definition.nodes["fusion_node"](state))["metadata"])
out = await definition.nodes["output_formatter_node"](state)

print("final answer:", out["messages"][-1].content)
```

What happens at each hop:

1. **`router`** calls `classify_modality()` on the last message (the `image/png` attachment hint wins) and stores `{"modality": "image", "confidence": ...}` under `metadata.mm.router`. A `force_modality` key in `metadata.mm` would override the classifier with confidence 1.0.
2. **Conditional edge** — `definition.conditional_edges["router"]` reads that stored decision and maps it through the modality table (`image → vision_node`, `audio → audio_node`, `video → video_node`, everything else → `text_node`).
3. **`vision_node`** resolves the media, awaits `vision_agent.analyze()`, and appends a `ModalContribution`-shaped dict (`modality`, `content`, `agent_id`, `confidence` — 0.9 for a real result, 0.0 when the agent fell back) to `metadata.mm.contributions`.
4. **`fusion_node`** rebuilds `ModalContribution` objects from the accumulated contributions and calls `MultimodalFusion.combine()`; the answer and the strategy actually used land under `metadata.mm.fusion`.
5. **`output_formatter_node`** renders the fused answer according to `metadata.mm.preferred_output` — plain text here; `json` would serialize the answer plus the `mm` metadata (minus the media itself) into the message payload.

Expected output: the router decision (`vision_node`) followed by the mocked caption as the final answer.

## Key API

| Symbol | Role |
|---|---|
| `build_multimodal_subgraph()` | Builds the pipeline `SubgraphDefinition`; accepts injected agents, `fusion`, `fusion_strategy`, `settings` |
| `register_multimodal_pipeline()` | Idempotently installs the subgraph in a `SubgraphRegistry` |
| `SubgraphDefinition` | Topology container: `entry_point`, `nodes`, `edges`, `conditional_edges` |
| `VisionResult` | Value object returned by `VisionAgent.analyze()` (`description`, `objects`, `ocr_text`, `model_used`) |
| `classify_modality()` | Heuristic modality classifier used by the `router` node |
| `MultimodalFusion` | Combines `ModalContribution`s via `moa` \| `moderator` \| `concat` strategies |
| `ModalContribution` | One modality's output handed to fusion (`modality`, `content`, `agent_id`, `confidence`) |
| `state["metadata"]["mm"]` | Namespaced multimodal state: `media`, `router`, `contributions`, `fusion`, `preferred_output` |

## Run it

```bash
uv run jupyter lab notebooks/multimodal_pipeline.ipynb
uv run python examples/multimodal_pipeline.py   # from the prismal repo
```

Runs entirely offline — no API keys required; all providers are stubbed or mocked.

## Related

- [Media ingestion into AgentState](multimodal_ingestion.md)
- [Modality router](multimodal/04_modality_router.md)
- [Multimodal fusion](multimodal/05_multimodal_fusion.md)
- [Vision agent](multimodal/01_vision_agent.md)
