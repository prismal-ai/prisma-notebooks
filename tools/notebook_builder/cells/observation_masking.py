from nbkit import header, md, code
CELLS = [
header("In-loop observation masking + `read_artifact` (Phase TKE4 · 3.20.0)",
       "prismal/agents/context_compaction.py",
       "🔑 Not required — synthetic tool history, in-memory artifact store",
       """
       Long ReAct loops are dominated by **old tool observations** that the model already used. With
       `context_compaction_strategy="mask_observations"`, the `ContextCompactor` moves those payloads into a scoped
       `ArtifactStorePort` and replaces them *in place* by a short stub — the tool call / tool result pairing stays
       valid, nothing is dropped. Since 3.20.0:

       * masking actually runs inside every agent's `react_loop` (an AST guard keeps the call wired);
       * the compactor and the tool-output budget share **one artifact store per process and backend** (bounded: 1,000 artifacts per tenant, 10,000 overall);
       * agents get a `read_artifact` tool bound to their own `(tenant, run_id)`, windowed to 8,000 characters with a next-start pointer — the stub says *"use read_artifact"* because without it models never re-read elided data (TKE9-04: 0/3).

       No standalone script exists in `examples/`; this notebook is written against `prismal/agents/context_compaction.py`
       and `prismal/agents/artifacts.py` (tests: `tests/unit/agents/test_mask_observations.py`, `test_read_artifact_injection.py`).
       """, "docs/token-efficiency.md",
       extra_rows=[("**Since**", "prismal 3.20.0")]),
md("## Setup · settings, store and a synthetic tool history\n\n`context_mask_keep_turns=2` keeps the observations of the last two turns verbatim; everything older is masked."),
code('''
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from prismal.agents.artifacts import InMemoryArtifactStore, make_read_artifact_tool
from prismal.agents.context_compaction import ContextCompactor
from prismal.core.config import Settings

settings = Settings(
    token_efficiency_enabled=True,
    context_compaction_enabled=True,
    context_compaction_strategy="mask_observations",
    context_mask_keep_turns=2,
    context_compaction_max_messages=3,
)
store = InMemoryArtifactStore(settings=settings)
compactor = ContextCompactor(settings=settings, artifact_store=store)

history = [
    HumanMessage(content="Find order 1042", id="h1"),
    AIMessage(content="", id="a1", tool_calls=[
        {"name": "search_orders", "args": {"q": "1042"}, "id": "c1", "type": "tool_call"}]),
    ToolMessage(content="order 1042: " + "line item; " * 900 + "total=EUR 318.40",
                id="t1", tool_call_id="c1", name="search_orders"),
    AIMessage(content="", id="a2", tool_calls=[
        {"name": "fetch_invoice", "args": {"id": "1042"}, "id": "c2", "type": "tool_call"}]),
    ToolMessage(content="invoice PDF text " * 300, id="t2", tool_call_id="c2", name="fetch_invoice"),
    HumanMessage(content="And the total?", id="h2"),
    AIMessage(content="Checking.", id="a3"),
]
print([(m.type, m.id, len(str(m.content))) for m in history])
'''),
md("## 1 · Mask the old observations\n\nThe full payload is stored **first**, then the `ToolMessage` is replaced by a stub with the same `id`, `tool_call_id` and `name`."),
code('''
should, reason = compactor.should_compact(history)
print(f"should_compact={should} ({reason})")

result = await compactor.mask_observations(history, run_id="run-7", tenant="acme")
print(f"compacted={result.compacted} strategy={result.strategy.value} dropped={result.messages_dropped}")
for msg in result.replaced:
    print(f"  {msg.id}: {msg.content}")
'''),
md("## 2 · Apply it through the LangGraph reducer\n\n`to_state_update()` returns messages with the *same ids*, so `add_messages` replaces them in place — checkpointed history shrinks through the normal reducer channel."),
code('''
update = compactor.to_state_update(result)
compacted = add_messages(history, update["messages"])
before = sum(len(str(m.content)) for m in history)
after = sum(len(str(m.content)) for m in compacted)
print([(m.id, len(str(m.content))) for m in compacted])
print(f"characters in history: {before} → {after}")
'''),
md("## 3 · The agent reads it back with `read_artifact`\n\n`react_loop` binds this tool to the loop's own scope. Reads are windowed (here 200 chars to keep the output short; 8,000 in the loop) and pass through the same sanitizer as the original tool output."),
code('''
artifact_id = str(result.replaced[0].content).split("artifact=")[1].split()[0]
read_artifact = make_read_artifact_tool(store, run_id="run-7", tenant="acme", max_chars=200)

print(read_artifact.invoke({"artifact_id": artifact_id}))
print("…")
print(read_artifact.invoke({"artifact_id": artifact_id, "start": 9900}))
'''),
md("## 4 · Scope isolation\n\nAnother run of the same tenant, or the same run id under another tenant, gets `artifact not found` — never an exception into the agent loop."),
code('''
print("other run   :", make_read_artifact_tool(store, run_id="run-8", tenant="acme").invoke({"artifact_id": artifact_id}))
print("other tenant:", make_read_artifact_tool(store, run_id="run-7", tenant="globex").invoke({"artifact_id": artifact_id}))
'''),
md("""
---

## Related settings

| Setting | Role |
|---|---|
| `context_compaction_strategy` | `truncate` · `summarize` · `mask_observations` |
| `context_mask_keep_turns` | Recent turns whose observations stay verbatim |
| `context_compaction_high_water` / `low_water` | Hysteresis band (compact once per band) |
| `tool_output_budget_enabled`, `tool_output_max_tokens_default` | Cap *new* tool outputs (see [`token_efficiency.ipynb`](token_efficiency.ipynb)) |

The truncate / summarize strategies of the same compactor are in [`loop_hardening.ipynb`](loop_hardening.ipynb).
"""),
]
