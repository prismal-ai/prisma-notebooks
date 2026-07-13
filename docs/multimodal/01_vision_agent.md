# VisionAgent — Image Analysis with an Injectable Vision LLM

> **Notebook:** [`notebooks/multimodal/01_vision_agent.ipynb`](../../notebooks/multimodal/01_vision_agent.ipynb) · **Example:** [`examples/multimodal/01_vision_agent.py`](https://github.com/prismal-ai/prismal/blob/main/examples/multimodal/01_vision_agent.py)

`VisionAgent` (SPEC-MM-AGT-001) is the image-understanding entry point of prismal's multimodal layer (Fase F). It receives an image as `bytes` or a `Path`, runs it through the `MediaValidator` security gate, delegates the description to a vision LLM, and optionally performs a second OCR pass — always returning a structured `VisionResult`. The multimodal layer is strictly opt-in: it is gated by `settings.multimodal_enabled` (default `False`), so the 26 text agents behave identically until you turn it on.

This demo exercises the agent's **factory-injection** design: instead of calling a real VLM, it injects a fake `vision_fn` and `ocr_fn`, so no Whisper, no vision model, no FFmpeg, and no API keys are needed — everything runs offline and deterministically. The same seam is what makes the agent unit-testable: business logic accepts callables, while the layered defaults (`_make_default_vision_fn`) wire the real `get_vision_llm` call lazily, and only when nothing was injected.

## What it demonstrates

- Building a small synthetic image corpus from arXiv cs.CV paper metadata (hand-rolled solid-color PNGs, no Pillow required).
- Injecting a fake `vision_fn` that maps image bytes back to an expected caption — the full `VisionAgent` pipeline runs with zero network calls.
- The `with_ocr=True` second pass through an injected `ocr_fn`, populating `VisionResult.ocr_text`.
- The `MediaValidator` gate rejecting a non-image blob by magic bytes, and the agent degrading gracefully (`used_fallback=True`) instead of raising.
- The `VisionResult` contract: `description`, `objects`, `ocr_text`, `model_used`, `used_fallback`.

## How it works

### Dataset: synthetic figures from arXiv cs.CV papers

The demo reads a local CSV of arXiv papers and, for each cs.CV entry, fabricates a minimal valid PNG plus the caption an "ideal" VLM should produce. The PNG generator writes the `IHDR`/`IDAT`/`IEND` chunks by hand:

```python
def _png_solid(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Generate a valid PNG (1 solid color) without Pillow. Demo only."""
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data)
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b""
    row = b"\x00" + bytes(rgb) * width
    for _ in range(height):
        raw += row
    idat = zlib.compress(raw, 9)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
```

The PNG signature matters: `MediaValidator.sniff()` recognizes formats by magic bytes, so these tiny images pass validation exactly like real photos would. If the CSV is missing, an embedded three-paper fallback keeps the notebook runnable anywhere.

### Injecting a fake vision function

`VisionAgent` takes `vision_fn: async (image, prompt) -> str`. The fake indexes the corpus by the first 16 bytes of each image and returns the pre-computed caption:

```python
def make_fake_vision_fn(corpus: list[PaperFigure]):
    """Vision-fn that looks at the first bytes to identify the paper."""
    by_signature = {fig.image_bytes[:16]: fig for fig in corpus}

    async def _vision(image, prompt: str) -> str:
        blob = image if isinstance(image, bytes) else Path(image).read_bytes()
        fig = by_signature.get(blob[:16])
        if fig is None:
            return f"[mock-vlm] cannot identify image · prompt={prompt!r}"
        return fig.expected_caption

    return _vision
```

In production, the layered default built by `_make_default_vision_fn` calls `get_vision_llm` and — critically — routes the prompt through `SecurePromptBuilder`, because image captions and user prompts are user-controlled content that must never be f-stringed into a template.

### Standard analysis, then OCR

The agent is constructed once with both fakes and `degrade_gracefully=True` (the default), then analyzed with and without OCR:

```python
agent = VisionAgent(
    vision_fn=make_fake_vision_fn(corpus),
    ocr_fn=make_fake_ocr_fn,
    degrade_gracefully=True,
)

result: VisionResult = await agent.analyze(fig.image_bytes, with_ocr=False)

result = await agent.analyze(
    corpus[0].image_bytes,
    prompt="What topic does this paper cover?",
    with_ocr=True,
)
print(result.ocr_text)   # "arXiv preprint · 2024" from the fake ocr_fn
```

When `with_ocr` is `None`, the agent falls back to `settings.vision_ocr_enabled`. An OCR failure under graceful degradation only logs a warning — the description still comes back; with `degrade_gracefully=False` it raises `VisionAgentError` instead.

### Security seam: validation before the model call

`MediaValidator.validate()` runs **before** any VLM call — magic-byte sniffing plus per-kind size limits — so junk never reaches the model or the network:

```python
bogus = b"this is not an image"
result = await agent.analyze(bogus)
assert result.used_fallback, "The agent must degrade to fallback"
print(result.description)   # "" — empty in fallback, no VLM call was made
```

Because `expected_kind=MediaKind.IMAGE` is passed, even a *valid* WAV or MP4 would be rejected here. This ordering (validate → sanitize → model) is a Fase F invariant: `MediaValidator` runs before `InputSanitizer.sanitize_media()`, and any text extracted from media (OCR output, captions) is treated as untrusted input downstream.

## Key API

| Symbol | Role |
|---|---|
| `VisionAgent` | Image analysis agent; constructor takes `vision_fn`, `ocr_fn`, `media_validator`, `degrade_gracefully`, `settings` (all keyword-only). |
| `VisionAgent.analyze(image, *, prompt=None, with_ocr=None)` | Validate → describe → optional OCR; returns `VisionResult`. |
| `VisionResult` | Frozen dataclass: `description`, `objects`, `ocr_text`, `model_used`, `used_fallback`. |
| `DetectedObject` | Optional object entry with `label`, `confidence`, normalized `bbox`. |
| `vision_fn` | Injectable `async (image, prompt) -> str`; default wires `get_vision_llm` + `SecurePromptBuilder` lazily. |
| `ocr_fn` | Injectable `async (image) -> str`; default is a second VLM pass with an OCR prompt. |
| `MediaValidator` | Magic-byte + size-limit gate run before any model call. |
| `MediaKind.IMAGE` | Expected-kind constraint passed to `validate()`. |
| `degrade_gracefully` | `True` (default): failures yield a fallback `VisionResult`; `False`: raise `VisionAgentError`. |
| `VisionAgentError` | Raised only when not degrading gracefully. |

## Run it

```bash
uv run jupyter lab notebooks/multimodal/01_vision_agent.ipynb
uv run python examples/multimodal/01_vision_agent.py   # from the prismal repo
```

No API keys required — the notebook injects fake vision/OCR callables and runs fully offline.

## Related

- [AudioAgent (voice-to-voice)](02_audio_agent.md)
- [VideoAgent](03_video_agent.md)
- [Multimodal pipeline subgraph](../multimodal_pipeline.md)
- [Multimodal ingestion](../multimodal_ingestion.md)
