# AG-UI 1.0 Event Translation

> **Notebook:** [`notebooks/agui_translate.ipynb`](../notebooks/agui_translate.ipynb) · **Example:** [`examples/agui_translate.py`](https://github.com/prismal-ai/prismal/blob/main/examples/agui_translate.py) · **Guide:** [`docs/protocols-2026.md`](https://github.com/prismal-ai/prismal/blob/main/docs/protocols-2026.md)

Phase PRT (3.17.0) added `prismal/events/agui.py`. It is a **pure** translator from LangGraph `astream(stream_mode=["messages", "updates", "custom"])` chunks to AG-UI 1.0 events. It has no HTTP code, and the AST guard `tests/unit/events/test_no_http_imports.py` keeps it that way. It never raises. The host (`prismal-server`) owns the transport.

## Mapping in the notebook

| `astream` item | AG-UI events |
|---|---|
| `updates` from a node | `STEP_STARTED` → `STATE_DELTA` (JSON Patch) → `STEP_FINISHED` (+ `CUSTOM` for routing metadata) |
| `messages` with `tool_call_chunks` | `TOOL_CALL_START` / `TOOL_CALL_ARGS` |
| `messages` with a `ToolMessage` | `TOOL_CALL_END` / `TOOL_CALL_RESULT` |
| `messages` with text chunks | `TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` … `TEXT_MESSAGE_END` |
| `custom` `{"type": "subagent"}` | `SUBAGENT_STARTED` / `SUBAGENT_FINISHED` / `SUBAGENT_ERROR` (3.20.0, emitted by Skynet workers and MoA proposers) |

`start_run()` emits `RUN_STARTED` and a `STATE_SNAPSHOT`. AG-UI 1.0 has no `USAGE` or `INTERRUPT` events, so `finish_run()` does two things:

- It puts usage in `RUN_FINISHED.usage[]` and also emits a `CUSTOM prismal.usage` event carrying the shared **12-key usage wire** (`RunSummary.to_usage_wire()`).
- It reports interrupts in `RUN_FINISHED.outcome`.

The usage wire is a contract shared with prismal-server and the SDK. Its fixture is `tests/fixtures/protocols/usage_wire.json`, so any change to the wire has to be made on both sides together.

## Same phase, not in this notebook

- A2A v0.3 + v1.0 dual-stack, with the version negotiated per request.
- MCP 2026-07-28 stateless client with a TTL tool cache and the tasks extension.
- The read-only host surface on `RuntimeContext`.

Wire names for all of these are pinned in `docs/adr/ADR-PRT-001-protocol-pins.md`.
