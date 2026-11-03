from nbkit import header, md, code
CELLS = [
header("AG-UI 1.0 event translation — fully offline (Phase PRT)", "examples/agui_translate.py",
       "🔑 Not required — the `astream` chunks are hand-built",
       """
       `prismal.events.agui` is a **pure translator**: it turns each item of
       `graph.astream(input, config, stream_mode=["messages", "updates", "custom"])` into AG-UI 1.0 events. It has no
       HTTP code (an AST guard enforces that) and never raises — the host (`prismal-server`) owns the transport.

       AG-UI 1.0 has no `USAGE` or `INTERRUPT` events, so usage travels in `RUN_FINISHED.usage[]` plus a
       `CUSTOM prismal.usage` event, and interrupts in `RUN_FINISHED.outcome`. Since 3.20.0, Skynet workers and MoA
       proposers emit `subagent` custom payloads that become `SUBAGENT_STARTED / FINISHED / ERROR`.

       In a real host:

       ```python
       ctx = AgUiContext(run_id=run_id, thread_id=thread_id)
       frames = [to_sse(e) for e in start_run(ctx, initial_state=state)]
       async for chunk in graph.astream(state, config, stream_mode=["messages", "updates", "custom"]):
           frames += [to_sse(e) for e in to_agui_events(chunk, ctx)]
       frames += [to_sse(e) for e in finish_run(ctx, summary=get_run_summary(thread_id))]
       ```
       """, "docs/protocols-2026.md"),
md("## Setup · imports and a fake `astream`"),
code('''
from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage

from prismal.budget.types import Usage
from prismal.events.agui import AgUiContext, finish_run, start_run, to_agui_events, to_sse
from prismal.monitoring.observability_types import RunSummary


def fake_astream() -> list[Any]:
    """What ``graph.astream(stream_mode=["messages", "updates", "custom"])`` would yield."""
    meta = {"langgraph_node": "agent"}
    return [
        ("updates", {"supervisor": {"next_agent": "agent", "metadata": {"routing": {"tier": "fast"}}}}),
        (
            "messages",
            (
                AIMessageChunk(
                    content="",
                    id="msg-1",
                    tool_call_chunks=[
                        {"id": "call-1", "name": "web_search", "args": '{"q": "ag-ui"}', "index": 0}
                    ],
                ),
                meta,
            ),
        ),
        ("messages", (ToolMessage(content="3 results", tool_call_id="call-1", id="tool-1"), meta)),
        ("custom", {"type": "subagent", "phase": "start", "id": "w1", "name": "skynet_worker"}),
        ("custom", {"type": "subagent", "phase": "end", "id": "w1", "result": "ok"}),
        ("messages", (AIMessageChunk(content="AG-UI is ", id="msg-2"), meta)),
        ("messages", (AIMessageChunk(content="an event protocol.", id="msg-2"), meta)),
        ("updates", {"agent": {"current_agent": "agent"}}),
    ]
'''),
md("## 1 · Translate chunk by chunk\n\nThe `AgUiContext` keeps the small amount of state the translator needs (open messages, open tool calls, steps)."),
code('''
ctx = AgUiContext(run_id="run-1", thread_id="thread-1")
events = start_run(ctx, initial_state={"current_agent": "supervisor"})
for chunk in fake_astream():
    produced = to_agui_events(chunk, ctx)
    print(f"{chunk[0]:8} -> {[e.type.value for e in produced]}")
    events.extend(produced)
'''),
md("## 2 · Finish the run with a usage summary\n\n`RunSummary.to_usage_wire()` is the shared 12-key usage wire (contract fixture shared with prismal-server and the SDK)."),
code('''
summary = RunSummary(
    run_id="run-1",
    session_id="thread-1",
    agent_name="supervisor",
    visited_nodes=["supervisor", "agent"],
    spans=[],
    tool_calls=[],
    usage=Usage(prompt_tokens=120, completion_tokens=30, calls=2, cost_source="table"),
    latency_ms=850.0,
)
events.extend(finish_run(ctx, summary=summary))
print("usage wire:", summary.to_usage_wire())
'''),
md("## 3 · Serialize to SSE frames"),
code('''
frames = [to_sse(e) for e in events]
print(Counter(e.type.value for e in events))
print()
print("".join(frames))
'''),
md("---\n\nThe same phase also ships the A2A v0.3 + v1.0 dual-stack, the MCP 2026-07-28 stateless wire with the tasks extension, and the read-only host surface on `RuntimeContext` (`budget_status`, `quota_windows`, `audit_page`, `decisions_report`, `routing_report`, …). Wire names are pinned in `docs/adr/ADR-PRT-001-protocol-pins.md`."),
]
