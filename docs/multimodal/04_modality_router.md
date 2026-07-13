# ModalityRouter — Deterministic Modality Classification

> **Notebook:** [`notebooks/multimodal/04_modality_router.ipynb`](../../notebooks/multimodal/04_modality_router.ipynb) · **Example:** [`examples/multimodal/04_modality_router.py`](https://github.com/prismal-ai/prismal/blob/main/examples/multimodal/04_modality_router.py)

The modality router (SPEC-MM-AGT-004) is the front door of prismal's opt-in Fase F multimodal layer (gated by `settings.multimodal_enabled`, default `False`): given an incoming message, decide whether it should go to the vision, audio, or video agent, to fusion (mixed media), or straight to the text path. `classify_modality()` is a pure, deterministic, LLM-free heuristic — attachments first, content blocks second, intent regex third — and `make_modality_router_node()` wraps it as a LangGraph node. An LLM fallback exists, but it is opt-in (`use_llm_fallback=True`) and only fires when the heuristics come up empty.

The demo needs no fakes at all in the usual sense, because the classifier itself makes no model calls: 18 labeled messages exercise every channel of the heuristic, and the router node runs against plain dict states. No Whisper, no VLM, no FFmpeg, and no API keys — everything is offline. Determinism-first routing is a deliberate design: classification is O(1), reproducible, and free, and the expensive LLM opinion is reserved for the genuinely ambiguous residue.

## What it demonstrates

- `classify_modality()` over 18 labeled cases spanning TEXT (ATIS queries), IMAGE, AUDIO, VIDEO, MIXED, and UNKNOWN, with per-case confidence scores.
- The three detection channels in priority order: attachment MIME types → multimodal content blocks (`image_url`, `input_audio`, `video`, …) → intent regex over the text.
- `make_modality_router_node()` routing states to `vision_agent | audio_agent | video_agent | fusion | text` and recording its decision under `state["metadata"]["mm"]["router"]`.
- The `force_modality` override: `state["metadata"]["mm"]["force_modality"]` bypasses classification entirely (confidence 1.0) — useful for tests and for letting users overrule the heuristic.
- The opt-in LLM fallback surface (`use_llm_fallback=True`), left off here so the demo stays fully deterministic.

## How it works

### A labeled corpus for every channel

Each case declares its text, an optional attachment MIME string, an optional content-block type, and the expected `Modality`:

```python
CASES = [
    ("atis_t1", "Show me all flights from Boston to Denver", None, None, Modality.TEXT),
    ("img_a1", "What's in this photo?", "image/png", None, Modality.IMAGE),
    ("img_b1", None, None, "image_url", Modality.IMAGE),
    ("img_r1", "Can you analyze the picture I uploaded yesterday?", None, None, Modality.IMAGE),
    ("aud_r1", "Please transcribe the call from this morning", None, None, Modality.AUDIO),
    ("vid_a1", "Summarize this clip", "video/mp4", None, Modality.VIDEO),
    ("mix_1", "Compare these", "image/png,audio/wav", None, Modality.MIXED),
    ("unk_1", "", None, None, Modality.UNKNOWN),
]
```

A helper builds a `HumanMessage` from each row, placing MIME types into `additional_kwargs["attachments"]` and block types into a list-style `content` — the two shapes real chat frontends produce.

### The classification cascade

`classify_modality(message)` inspects the message in a strict order and returns a `ModalityClassification` whose confidence reflects the strength of the evidence:

1. **Attachments** — `additional_kwargs["attachments"][i].mime_type` mapped by prefix (`image/`, `audio/`, `video/`). One modality → confidence 0.95; more than one → `MIXED` at 0.9.
2. **Content blocks** — list-style `content` entries whose `type` is `image_url`, `image`, `input_audio`, `audio`, `video`, or `video_url` merge into the same modality set.
3. **Intent regex** — text-only messages match patterns such as `transcribe|voice|audio|speech` → AUDIO, `video|clip|footage` → VIDEO, `image|picture|photo|screenshot` → IMAGE, at confidence 0.6.
4. **Defaults** — non-empty text → `TEXT` (0.7); nothing at all → `UNKNOWN` (0.0).

Hard evidence (an actual attachment) always beats textual hints, and the regex channel means "transcribe the call from this morning" routes to the audio agent even with no attachment present.

### The LangGraph node

`make_modality_router_node()` adapts the classifier to graph state. Its output carries both the routing decision and an audit trail of *why*:

```python
router = make_modality_router_node(use_llm_fallback=False)

state = {"messages": [msg], "metadata": {}}
result = await router(state)
# result["next"] ∈ {"vision_agent", "audio_agent", "video_agent", "fusion", "text"}
# result["metadata"]["mm"]["router"] == {
#     "modality": ..., "confidence": ..., "detected_attachments": [...],
#     "used_fallback_llm": False,
# }
```

The decision metadata lives under `state["metadata"]["mm"]`, the namespace that isolates all multimodal state from the rest of `AgentState` — the same convention used for path-based media descriptors, which keeps raw bytes out of checkpointed state.

### Forcing a modality

The override consults `state["metadata"]["mm"]["force_modality"]` before any classification:

```python
msg = HumanMessage(content="random text without intent")
state = {
    "messages": [msg],
    "metadata": {"mm": {"force_modality": Modality.AUDIO.value}},
}
result = await router(state)
assert result["next"] == "audio_agent"   # heuristics never ran
```

Two edge behaviors round out the node's contract. A state with no messages at all routes to `text` with confidence 1.0 — an empty turn has nothing to classify, and the text path is the safe default. And the demo's second section groups all 18 cases by their destination node, making the routing table visible at a glance:

```python
targets = {"vision_agent": [], "audio_agent": [], "video_agent": [], "fusion": [], "text": []}
for case_id, text, mime, block, _ in CASES:
    msg = build_message(text, mime, block)
    state = {"messages": [msg], "metadata": {}}
    result = await router(state)
    targets[result["next"]].append(case_id)
```

With `use_llm_fallback=True`, an `UNKNOWN` classification would instead trigger one multimodal LLM call — routed through `SecurePromptBuilder`, since the message text is user-controlled content — and the result is flagged with `used_fallback_llm=True` at confidence 0.5, so downstream consumers can tell heuristic decisions from model-assisted ones.

## Key API

| Symbol | Role |
|---|---|
| `Modality` | `StrEnum`: `TEXT`, `AUDIO`, `IMAGE`, `VIDEO`, `MIXED`, `UNKNOWN`. |
| `classify_modality(message, *, settings=None)` | Pure heuristic classifier; attachments → blocks → intent regex → TEXT/UNKNOWN. |
| `ModalityClassification` | Frozen dataclass: `modality`, `confidence`, `detected_attachments`, `used_fallback_llm`. |
| `make_modality_router_node(*, use_llm_fallback=False, settings=None)` | Builds the async LangGraph node returning `{"next", "metadata"}`. |
| `_MODALITY_TO_NODE` mapping | IMAGE→`vision_agent`, AUDIO→`audio_agent`, VIDEO→`video_agent`, MIXED→`fusion`, TEXT→`text`. |
| `state["metadata"]["mm"]["force_modality"]` | Per-request override; wins over classification with confidence 1.0. |
| `state["metadata"]["mm"]["router"]` | Where the node records modality, confidence, attachments, and fallback usage. |
| `use_llm_fallback` | Opt-in: on `UNKNOWN`, ask a multimodal LLM (via `SecurePromptBuilder`) for one word: text/audio/image/video. |

## Run it

```bash
uv run jupyter lab notebooks/multimodal/04_modality_router.ipynb
uv run python examples/multimodal/04_modality_router.py   # from the prismal repo
```

No API keys required — the classifier is deterministic and LLM-free; the notebook runs fully offline.

## Related

- [VisionAgent](01_vision_agent.md)
- [AudioAgent (voice-to-voice)](02_audio_agent.md)
- [MultimodalFusion](05_multimodal_fusion.md)
- [Multimodal pipeline subgraph](../multimodal_pipeline.md)
