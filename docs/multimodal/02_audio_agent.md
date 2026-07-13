# AudioAgent — Voice-to-Voice (STT → Reason → TTS)

> **Notebook:** [`notebooks/multimodal/02_audio_agent.ipynb`](../../notebooks/multimodal/02_audio_agent.ipynb) · **Example:** [`examples/multimodal/02_audio_agent.py`](https://github.com/prismal-ai/prismal/blob/main/examples/multimodal/02_audio_agent.py)

`AudioAgent` (SPEC-MM-AGT-002) implements the classic voice-assistant cascade: validate the audio, transcribe it (STT), reason over the transcript with an LLM, and optionally synthesize a spoken reply (TTS). It belongs to prismal's opt-in Fase F multimodal layer — gated by `settings.multimodal_enabled` (default `False`), so nothing changes for text-only deployments until you flip the switch.

The demo runs the whole pipeline **offline**: it injects a fake `STTClient`, a fake `TTSClient`, and a deterministic `reason_fn`, so no Whisper, no ElevenLabs, no provider SDKs, and no API keys are involved. That is the point of the factory-injection design — the agent's control flow (validation, span instrumentation, graceful degradation) is fully exercised while every backend is a test double; in production the same constructor resolves `get_stt()` / `get_tts()` lazily and builds a `SecurePromptBuilder`-guarded LLM reasoning function.

## What it demonstrates

- The full STT → reason → TTS turn over ATIS-style flight-assistant utterances (4 embedded samples, English and Spanish).
- Generating valid silent WAV files with the stdlib `wave` module — enough to pass `MediaValidator`'s magic-byte and duration checks.
- Injecting `stt_client`, `tts_client`, and `reason_fn` so the agent runs without any speech backend.
- Batch processing with `with_tts=False` — the cheaper mode for text chatbots that only need transcripts and replies.
- `MediaValidator` rejecting a non-audio blob before STT is ever called, with the agent degrading to an empty `AudioResult`.

## How it works

### Dataset: ATIS-style voice intents

ATIS (Air Travel Information System) is the canonical voice-command dataset. The demo embeds four utterances, each pairing a transcript with the reply the assistant should give. Since real recordings aren't needed, each utterance is represented by a silent PCM WAV of a distinct duration:

```python
def _make_silence_wav(duration_s: float = 1.0, sample_rate: int = 16_000) -> bytes:
    """Generate a silent PCM WAV (valid for MediaValidator)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        n_samples = int(duration_s * sample_rate)
        w.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()
```

The WAV header (`RIFF....WAVE`) is exactly what `MediaValidator.sniff()` looks for, and — because WAV duration is cheaply computable with the stdlib — the validator also enforces `max_audio_duration_s` on these blobs.

### Fake STT and TTS clients

The fakes honor the real provider protocols (`STTClient.transcribe` returning `STTResult`, `TTSClient.synthesize` returning `TTSResult`), so the agent cannot tell them apart from Whisper or a cloud TTS. The STT fake maps each WAV's 44-byte header to its dataset entry:

```python
class FakeSTT:
    """STTClient mock that maps audio -> transcript by byte hash."""

    def __init__(self, by_signature: dict[bytes, dict]) -> None:
        self._by_signature = by_signature

    async def transcribe(self, audio, *, language=None, prompt=None) -> STTResult:
        blob = audio if isinstance(audio, bytes) else Path(audio).read_bytes()
        sample = self._by_signature.get(blob[:44]) or {"transcript": "[unknown audio]", "language": "en"}
        text = sample["transcript"]
        return STTResult(
            text=text,
            language=language or sample["language"],
            segments=[STTSegment(start_s=0.0, end_s=1.0, text=text)],
            provider_used="mock-whisper",
        )
```

The fake TTS synthesizes a silence WAV whose duration scales with the reply length, returning a complete `TTSResult(audio, mime_type, provider_used, duration_s)`.

### The full turn and the batch mode

One agent handles both interaction styles. `with_tts=True` produces spoken output; `with_tts=False` skips synthesis entirely, which is what a text chatbot wants:

```python
agent = AudioAgent(
    stt_client=FakeSTT(by_signature),
    tts_client=FakeTTS(),
    reason_fn=make_reason_fn(by_transcript),
    degrade_gracefully=True,
)

result: AudioResult = await agent.process(wav, language="en", with_tts=True)
print(result.transcript, result.response_text, result.response_mime)

for i, ut in enumerate(ATIS_UTTERANCES):
    wav = _make_silence_wav(duration_s=0.5 + i * 0.25)
    result = await agent.process(wav, language=ut["language"], with_tts=False)
```

Each stage (`mm.audio.validate`, `mm.audio.stt`, `mm.audio.reason`, `mm.audio.tts`) runs inside its own OTel span. Failures degrade per stage: an STT failure returns an empty result, a reasoning failure still preserves the transcript and STT provider in the fallback, and a TTS failure under graceful degradation just logs a warning and returns the text-only reply.

### Security seam: validate first, sanitize transcripts always

`MediaValidator.validate(blob, expected_kind=MediaKind.AUDIO)` runs before STT — magic bytes, size ceiling, and WAV duration — so garbage never reaches a speech backend:

```python
result = await agent.process(b"not a wav file")
print(result.transcript)     # ''
print(result.response_text)  # ''  — the agent degraded without calling STT
```

Downstream, the transcript is user-controlled content: the default `reason_fn` routes it through `SecurePromptBuilder` rather than interpolating it into a prompt template. That rule holds for any custom `reason_fn` you write — STT output is untrusted input, exactly like typed text.

## Key API

| Symbol | Role |
|---|---|
| `AudioAgent` | Voice-to-voice agent; constructor takes `stt_client`, `tts_client`, `reason_fn`, `media_validator`, `degrade_gracefully`, `settings`. |
| `AudioAgent.process(audio, *, state=None, language=None, with_tts=False)` | Runs validate → STT → reason → optional TTS; returns `AudioResult`. |
| `AudioResult` | Frozen dataclass: `transcript`, `response_text`, `response_audio`, `response_mime`, `stt_provider_used`, `tts_provider_used`, `duration_s`. |
| `STTClient` / `STTResult` / `STTSegment` | STT protocol and result types from `prismal.providers.stt`; default resolved lazily via `get_stt()`. |
| `TTSClient` / `TTSResult` | TTS protocol and result from `prismal.providers.tts`; default resolved lazily via `get_tts()`. |
| `reason_fn` | Injectable `async (transcript, state) -> str`; default is a provider LLM call through `SecurePromptBuilder`. |
| `MediaValidator` | Pre-STT gate: magic bytes, `max_audio_bytes`, WAV duration ceiling. |
| `degrade_gracefully` | `True` (default): failures yield fallback results; `False`: raise `AudioAgentError`. |
| `AudioAgentError` | Raised only when not degrading gracefully. |

## Run it

```bash
uv run jupyter lab notebooks/multimodal/02_audio_agent.ipynb
uv run python examples/multimodal/02_audio_agent.py   # from the prismal repo
```

No API keys required — the notebook injects fake STT/TTS clients and a deterministic reasoning function, running fully offline.

## Related

- [VisionAgent](01_vision_agent.md)
- [VideoAgent](03_video_agent.md)
- [ModalityRouter](04_modality_router.md)
- [Multimodal pipeline subgraph](../multimodal_pipeline.md)
