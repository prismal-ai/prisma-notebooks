# VideoAgent — Frames + Transcript + Fusion

> **Notebook:** [`notebooks/multimodal/03_video_agent.ipynb`](../../notebooks/multimodal/03_video_agent.ipynb) · **Example:** [`examples/multimodal/03_video_agent.py`](https://github.com/prismal-ai/prismal/blob/main/examples/multimodal/03_video_agent.py)

`VideoAgent` (SPEC-MM-AGT-003) is the composite agent of prismal's opt-in Fase F multimodal layer (gated by `settings.multimodal_enabled`, default `False`). Summarizing a video means orchestrating three sub-pipelines: extract sample frames (FFmpeg — always inside `SandboxExecutor` in production), describe each frame in parallel through a `VisionAgent`, transcribe the audio track, and fuse everything into a single summary. Every stage is an injected callable — `frame_extractor_fn`, `transcribe_fn`, `fusion_fn` — so the agent's control flow is testable without FFmpeg, a VLM, Whisper, or API keys.

This demo takes that seam to its logical end: three synthetic ActivityNet-style clips, a hand-built "MP4" that satisfies the magic-byte check, and mocked collaborators for every stage. The whole thing runs offline and deterministically, yet exercises the exact code paths a real deployment uses — validation, parallel frame description with `asyncio.gather`, per-stage graceful degradation, and the `VideoResult` contract.

## What it demonstrates

- Three synthetic clips (cooking, skateboarding, guitar lesson) with pre-defined frame captions and transcripts, mirroring ActivityNet Captions' `(start, end, sentence)` structure.
- A minimal fake MP4 — just a `ftyp` box at offset 4 — that passes `MediaValidator.sniff()`.
- Mock collaborators for all three injection points: frame extractor (writes tiny PNGs to a temp dir), `VisionAgent` with a filename-aware `vision_fn`, `transcribe_fn`, and a deterministic label-and-concatenate `fusion_fn`.
- Timestamped `FrameDescription` entries derived from the sampling `fps`, plus the fused summary.
- Validation rejecting a non-video blob: the agent returns an empty `VideoResult` without extracting a single frame.

## How it works

### Dataset and the fake MP4

Each clip declares its `fps`, per-frame captions, and an audio transcript. The video files themselves only need to *look like* MP4s to the validator:

```python
def _make_fake_mp4() -> bytes:
    """Minimal MP4: the detector looks for 'ftyp' at offset 4."""
    box = b"ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    return struct.pack(">I", len(box) + 4) + box + b"\x00" * 256
```

`MediaValidator` deliberately keeps video probing cheap: it checks magic bytes and size here, while real duration enforcement is left to `ffprobe` inside the sandbox — parsing containers in-process would defeat the isolation model.

### Injecting the three pipeline stages

The frame extractor mock writes N valid PNGs named `{clip}_frame_{i}.png` and returns their paths — the same contract as the production extractor, which shells out to FFmpeg via `SandboxExecutor` (never in-process):

```python
async def _extract(video: Path, fps: float, max_frames: int) -> list[Path]:
    clip_id = video.stem
    n_frames = min(frames_per_clip.get(clip_id, 3), max_frames)
    out: list[Path] = []
    for i in range(n_frames):
        p = out_dir / f"{clip_id}_frame_{i:03d}.png"
        p.write_bytes(_png(i * 30 % 255))
        out.append(p)
    return out
```

The vision stage reuses the real `VisionAgent`, itself carrying an injected `vision_fn` that decodes the clip id and frame index from the file name and returns the matching caption. Composition works because each layer accepts callables — the fake bottoms out exactly where the network call would be. Transcription and fusion follow the same shape:

```python
def make_fusion_fn():
    """Deterministic fusion: concat audio + frames with labels."""

    async def _fuse(transcript: str, frames: list[FrameDescription]) -> str:
        lines = [f"AUDIO: {transcript}"] if transcript else []
        for fr in frames:
            lines.append(f"FRAME {fr.timestamp_s:.1f}s: {fr.description}")
        return "\n".join(lines)

    return _fuse
```

In production, the default `transcribe_fn` extracts the audio track with sandboxed FFmpeg and hands it to an `AudioAgent`; the default `fusion_fn` calls a multimodal LLM through `SecurePromptBuilder` — transcripts and frame captions are user-controlled content and never get f-stringed into a prompt.

### Summarizing each clip

```python
agent = VideoAgent(
    vision_agent=make_vision_agent(captions_by_clip),
    frame_extractor_fn=make_frame_extractor(tmp_dir, frames_per_clip),
    transcribe_fn=make_transcribe_fn(transcripts_by_clip),
    fusion_fn=make_fusion_fn(),
    degrade_gracefully=True,
)

result: VideoResult = await agent.summarize(video_path, fps=clip["fps"], max_frames=10)
for fr in result.frame_descriptions:
    print(f"t={fr.timestamp_s:.1f}s · {fr.description}")
print(result.summary)
```

Frames are described concurrently (`asyncio.gather` with `return_exceptions=True` — one bad frame is logged and skipped, not fatal). Timestamps are computed as `index / fps`, so a 0.5 fps guitar lesson yields frames at 0.0s, 2.0s, 4.0s. When `fps` or `max_frames` are omitted, `settings.video_sample_fps` and `settings.max_frames_per_video` apply. Transcription and fusion each degrade independently: under `degrade_gracefully=True` a failed stage contributes an empty string instead of aborting the pipeline.

### Security seam: validation gates the whole pipeline

`MediaValidator.validate(video, expected_kind=MediaKind.VIDEO)` runs before any extraction. A garbage file short-circuits everything:

```python
bogus.write_bytes(b"garbage")
result = await agent.summarize(bogus, fps=1.0, max_frames=3)
assert result.summary == "" and result.total_frames_processed == 0
```

Two invariants worth repeating: FFmpeg runs **only** inside `SandboxExecutor` (both frame extraction and audio extraction in the defaults), and in graph deployments media reaches the agent as a path-based descriptor under `state["metadata"]["mm"]` — raw bytes never enter checkpointed state.

## Key API

| Symbol | Role |
|---|---|
| `VideoAgent` | Composite video pipeline; constructor takes `vision_agent`, `audio_agent`, `frame_extractor_fn`, `transcribe_fn`, `fusion_fn`, `media_validator`, `degrade_gracefully`, `settings`. |
| `VideoAgent.summarize(video, *, fps=None, max_frames=None)` | Validate → extract → describe → transcribe → fuse; returns `VideoResult`. |
| `VideoResult` | Frozen dataclass: `transcript`, `frame_descriptions`, `summary`, `total_frames_processed`, `duration_s`. |
| `FrameDescription` | Per-frame entry: `frame_index`, `timestamp_s`, `description`. |
| `frame_extractor_fn` | Injectable `async (video, fps, max_frames) -> list[Path]`; default runs FFmpeg inside `SandboxExecutor`. |
| `transcribe_fn` | Injectable `async (video) -> str`; default extracts audio via sandboxed FFmpeg then calls `AudioAgent.process`. |
| `fusion_fn` | Injectable `async (transcript, frames) -> str`; default is a multimodal LLM call through `SecurePromptBuilder`. |
| `VisionAgent` | Describes each sampled frame (parallelized); built lazily if not injected. |
| `MediaValidator` | Pre-pipeline gate (`MediaKind.VIDEO`); magic bytes + size, video duration deferred to sandboxed `ffprobe`. |
| `VideoAgentError` | Raised only when `degrade_gracefully=False`. |

## Run it

```bash
uv run jupyter lab notebooks/multimodal/03_video_agent.ipynb
uv run python examples/multimodal/03_video_agent.py   # from the prismal repo
```

No API keys required — the notebook injects fake frame-extraction, vision, transcription, and fusion callables and runs fully offline.

## Related

- [VisionAgent](01_vision_agent.md)
- [AudioAgent (voice-to-voice)](02_audio_agent.md)
- [MultimodalFusion](05_multimodal_fusion.md)
- [Multimodal pipeline subgraph](../multimodal_pipeline.md)
