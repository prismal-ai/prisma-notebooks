# Media Ingestion into AgentState and Pipeline Registration

> **Notebook:** [`notebooks/multimodal_ingestion.ipynb`](../notebooks/multimodal_ingestion.ipynb) · **Example:** [`examples/multimodal_ingestion.py`](https://github.com/prismal-ai/prismal/blob/main/examples/multimodal_ingestion.py)

`ingest_media()` is the single boundary where an incoming attachment becomes part of `AgentState`. This walkthrough creates a real, decodable 1x1 PNG and pushes it through the full ingestion contract — validate magic bytes and size (`MediaValidator`) → sanitize (EXIF strip via `InputSanitizer.sanitize_media`) → spill the bytes to a content-addressed file → audit hash-only → record a *path-based* descriptor under `state["metadata"]["mm"]["media"]` — and then registers the multimodal pipeline subgraph in a fresh `SubgraphRegistry` with `multimodal_enabled=True`.

The multimodal layer (Fase F) is **opt-in**: `settings.multimodal_enabled` defaults to `False`, and every collaborator of `ingest_media` (validator, sanitizer, audit logger, settings) is injectable. Here a console-printing audit stand-in replaces the JSONL `AuditLogger`, so the whole demo runs offline — no Whisper, no VLM, no FFmpeg, and no API keys.

## What it demonstrates

- The five-step ingestion contract: validate → sanitize → spill → audit → descriptor.
- Why checkpointed state carries a *path*, never raw bytes: LangGraph checkpoints all of `AgentState` on every super-step, so bytes in state would bloat the database and contradict the hash-only audit policy.
- Content-addressed spill layout (`<workspace>/<session_id>/<sha256>.<ext>`) and cheap per-session cleanup with `cleanup_session_media()`.
- Hash-first auditing: `log_media()` records SHA-256, modality, and size — never content.
- Registering `multimodal_pipeline` in an isolated `SubgraphRegistry` and inspecting the resulting `SubgraphDefinition`.

## How it works

### A real PNG and a duck-typed audit stand-in

The fixture must be a *complete* PNG, not just a header: its `\x89PNG\r\n\x1a\n` magic bytes satisfy `MediaValidator`'s sniffing, and because it decodes cleanly the EXIF-strip re-encode inside `InputSanitizer.sanitize_media` works when Pillow is installed.

```python
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)

SESSION_ID = "sess-mm-demo"


class _ConsoleAudit:
    """AuditLogger stand-in — prints the hash-first record instead of JSONL I/O."""

    def log_media(self, action: str, **fields: Any) -> None:
        print(f"  [audit] media {action}: {fields}")
```

`ingest_media` only calls `audit.log_media(...)`, so any object with that method works — the same duck-typing seam the test suite uses.

### Ingesting: validate → sanitize → spill → audit → descriptor

A temporary directory serves as the media workspace, and `Settings(multimodal_enabled=True, media_workspace=...)` is passed explicitly rather than read from the environment:

```python
workspace = Path(tempfile.mkdtemp(prefix="prismal_media_demo_"))
settings = Settings(multimodal_enabled=True, media_workspace=str(workspace))
state = create_initial_state(session_id=SESSION_ID)

ingest_media(
    state,
    _PNG_1X1,
    kind=MediaKind.IMAGE,
    source="example:generated",
    workspace=workspace,
    preferred_output="text",
    audit=_ConsoleAudit(),
    settings=settings,
)
```

Internally, `ingest_media` executes the security seams in a fixed order:

1. **Validate** — `MediaValidator.validate()` sniffs magic bytes and enforces size/duration limits; a failure raises `MediaValidationError` before anything touches disk. Passing `kind=None` would auto-detect the kind instead.
2. **Sanitize** — `InputSanitizer.sanitize_media()` runs *after* validation, re-encoding images to strip EXIF metadata (pass-through for other kinds).
3. **Spill** — the cleaned bytes are hashed (SHA-256) and written to `<workspace>/<session_id>/<sha256>.<ext>`. Before the write, `ActionInterceptor.check_media_op("write", target, workspace_root=...)` confirms the target stays inside the workspace — path confinement is enforced even for internally computed paths.
4. **Audit** — `log_media("ingested", sha256=..., modality=..., size_bytes=..., duration_s=...)` records the hash and modality, never the content.
5. **Descriptor** — a dict with `uri`, `kind`, `mime`, `sha256`, `source`, and `bytes_len` is appended to `state["metadata"]["mm"]["media"]`, and `primary_media_index` points at it.

### Inspecting the path-based descriptor

```python
mm = state["metadata"]["mm"]
descriptor = mm["media"][mm["primary_media_index"]]
print(json.dumps(descriptor, indent=2))
spilled = Path(descriptor["uri"])
print(f"spilled file exists: {spilled.is_file()} ({spilled.stat().st_size} bytes)")
# Only the path travels in checkpointed state — never the raw bytes.
```

This is the crux of the design: modal agents accept `bytes | Path`, so a path works everywhere, while the checkpointer only ever serializes a small JSON descriptor. The content-addressed filename also deduplicates identical uploads within a session for free.

### Registering the multimodal pipeline

With ingestion done, the demo registers the subgraph the supervisor would mount when `multimodal_enabled=True` — in a *fresh* registry, so nothing global is touched:

```python
registry = SubgraphRegistry()
register_multimodal_pipeline(registry, settings=settings, fusion_strategy="concat")
definition = registry.get("multimodal_pipeline")
print(f"Registered subgraphs: {registry.list()}")
print(f"Entry point: {definition.entry_point}")
print(f"Nodes: {list(definition.nodes)}")
```

`register_multimodal_pipeline` is idempotent (a second call is a no-op) and forwards build kwargs to `build_multimodal_subgraph`; the `concat` fusion strategy keeps the pipeline deterministic and LLM-free. The printed definition shows the full topology: `router` as entry point, then `vision_node`, `audio_node`, `video_node`, `text_node`, `fusion_node`, and `output_formatter_node`.

### Per-session cleanup

```python
cleanup_session_media(SESSION_ID, workspace=workspace)
print(f"After cleanup_session_media: spilled file exists = {spilled.is_file()}")
```

Because spilled files live under one directory per session, cleanup is a single `rmtree` — no bookkeeping table required. The final print confirms the file is gone.

## Key API

| Symbol | Role |
|---|---|
| `ingest_media()` | The ingestion boundary: validate → sanitize → spill → audit → descriptor; mutates and returns `state` |
| `cleanup_session_media()` | Removes all spilled media for one session (no-op if none exists) |
| `MediaValidator` / `MediaKind` | Magic-byte and size/duration validation; `MediaKind.IMAGE` declares the expected kind |
| `InputSanitizer.sanitize_media()` | EXIF strip for images (runs after validation), pass-through for other kinds |
| `ActionInterceptor.check_media_op()` | Path-confinement gate checked before the spill write |
| `AuditLogger.log_media()` | Hash-first audit record (SHA-256 + modality + size, never content) |
| `create_initial_state()` | Builds a fresh `AgentState` bound to a `session_id` |
| `register_multimodal_pipeline()` | Idempotently installs the pipeline `SubgraphDefinition` in a registry |
| `SubgraphRegistry` | Named store of subgraph definitions (`register_sync`, `get`, `list`) |
| `Settings(multimodal_enabled=True, media_workspace=...)` | Explicit opt-in configuration, injected rather than read from `.env` |

## Run it

```bash
uv run jupyter lab notebooks/multimodal_ingestion.ipynb
uv run python examples/multimodal_ingestion.py   # from the prismal repo
```

Runs entirely offline with injected fakes — no API keys required.

## Related

- [Multimodal pipeline subgraph end-to-end](multimodal_pipeline.md)
- [Vision agent](multimodal/01_vision_agent.md)
- [Modality router](multimodal/04_modality_router.md)
- [Multimodal RAG](rag/09_multimodal_rag.md)
