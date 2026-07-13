# Memory management: short-term, transcript, and long-term stores

> **Notebook:** [`notebooks/memory_management.ipynb`](../notebooks/memory_management.ipynb) · **Example:** [`examples/memory_management.py`](https://github.com/prismal-ai/prismal/blob/main/examples/memory_management.py)

prismal's memory layer sits beside the LangGraph supervisor state machine and gives agents three
complementary kinds of recall: a bounded in-session message buffer (`ShortTermMemory`), an
append-only Markdown transcript per session (`ConversationHistory`), and a cross-session fact
store (`LongTermMemory`) that persists to SQLite plus a vector store and redacts secrets before
anything touches disk. In production the vector side is ChromaDB, but the store is typed against
the `VectorStorePort` protocol (Fase Z), so this walkthrough swaps in a tiny in-memory keyword
index and runs the whole subsystem offline — no embeddings, no network, no LLM.

Reach for this notebook when you want to understand what each memory tier is responsible for,
how session scoping works (AC-011-2/-3), and how the redaction (AC-011-5) and retention
(AC-011-6) guarantees behave before you wire the layer into a real deployment.

## What it demonstrates

- `ShortTermMemory` as a thread-safe FIFO buffer: adding six turns to a four-slot buffer evicts
  the two oldest messages automatically.
- `ConversationHistory` writing an append-only Markdown transcript with YAML front-matter that
  tracks every channel (`cli`, `telegram`, ...) the session spoke through.
- `LongTermMemory` round-tripping facts through SQLite + a vector store, with automatic secret
  redaction before persistence and per-entry expiry timestamps.
- Session-scoped versus cross-session recall — the same query answered with and without a
  `session_id` filter.
- Substituting the default Chroma backend with any object that satisfies `VectorStorePort`,
  including its score contract (`score in [0, 1]`, higher = more relevant).

## How it works

### An in-memory `VectorStorePort` double

`LongTermMemory` never constructs its own vector backend; it accepts anything conforming to the
port. The notebook's `KeywordVectorStore` implements the five port methods with token-overlap
scoring — deterministic, dependency-free, and honouring the port's score contract:

```python
class KeywordVectorStore:
    def similarity_search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Rank stored documents by query-token overlap, descending."""
        q_tokens = self._tokens(query)
        if not q_tokens:
            return []
        scored = []
        for doc in self._docs:
            overlap = len(q_tokens & self._tokens(doc.page_content))
            score = round(overlap / len(q_tokens), 4)
            if score > 0.0:
                scored.append((doc, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
```

This is the same injection seam a test suite or an alternative backend (LanceDB, Qdrant,
pgvector) would use — the memory code cannot tell the difference.

### 1. Short-term memory: the bounded session buffer

`ShortTermMemory` wraps a `collections.deque` with `maxlen` semantics behind a lock, so when the
buffer is full the oldest message silently drops out (FIFO). The demo makes the eviction visible
by oversubscribing a four-slot buffer:

```python
mem = ShortTermMemory(max_messages=4)
turns = [
    HumanMessage(content="Hi, I'm setting up the RAG pipeline."),
    AIMessage(content="Great — which vector store backend do you want?"),
    HumanMessage(content="Let's start with the Chroma default."),
    AIMessage(content="Chroma it is. Collection name?"),
    HumanMessage(content="Use 'docs_v1' as the collection name."),
    AIMessage(content="Done: collection 'docs_v1' is configured."),
]
for msg in turns:
    mem.add(msg)
```

After the loop, `len(mem)` is 4 and `get_all()` returns only the last four turns in
chronological order; `get_recent(2)` slices the newest exchange, and `clear()` empties the
buffer. Everything is a `BaseMessage`, so the buffer plugs straight into LangGraph message flows.

### 2. Conversation history: the append-only transcript

`ConversationHistory` writes one Markdown file per session (`{base_dir}/{session_id}.md`),
created lazily with YAML front-matter on the first `append()`. Each turn records a timestamp,
role, and source channel; when a new channel appears, the front-matter `channels` list is
updated. Write errors are logged and swallowed — history is best-effort and never breaks a turn.

```python
history = ConversationHistory(SESSION_A, base_dir=base_dir)
history.append("User", "Remember that I prefer dark mode.", channel="cli")
history.append("Agent", "Noted — dark mode preference saved.", channel="cli")
history.append("User", "Also enable compact layout.", channel="telegram")

print(f"Transcript file: {history.path().name} (in a temp dir)")
for line in history.read().splitlines():
    print(f"  {line}")
```

The printed file shows `channels: [cli, telegram]` in the front-matter followed by one dated
`## timestamp · role · channel` section per turn.

### 3. Long-term memory: SQLite + vector store with redaction

`LongTermMemory` is the cross-session tier. `save()` redacts known secret patterns (Anthropic,
OpenAI, Google, GitHub, AWS keys, generic `api_key=`/`token=` shapes) *before* the content is
written to either SQLite or the vector store, then stamps an `expires_at` derived from
`retention_days`. The demo saves a fact that embeds a fake OpenAI-style key and proves the
secret never reaches disk:

```python
store = KeywordVectorStore()
mem = LongTermMemory(db_path=base_dir / "memory.db", vector_store=store, retention_days=30)

fake_key = "sk-" + "a1B2c3D4" * 6  # matches the OpenAI-style pattern
for session_id, content in facts:
    entry_id = await mem.save(content, session_id=session_id)

recalled = await mem.recall("staging deploy credential", session_id=SESSION_A)
for entry in recalled:
    assert fake_key not in entry.content, "secret must be redacted"
```

`recall()` then demonstrates scoping: with `session_id=SESSION_A` the query about a Python
developer finds nothing (that fact belongs to session B), while `session_id=None` searches
across all sessions (AC-011-2). Each returned `MemoryEntry` carries its `expires_at`, and the
maintenance surface is exercised last:

```python
expired = await mem.expire()                    # prune stale entries (AC-011-6)
cleared = await mem.clear(session_id=SESSION_A) # drop one session's facts (AC-011-4)
remaining = await mem.recall(query, session_id=None)  # only session B survives
```

`clear()` removes rows from SQLite and the matching documents from the vector store, keeping
both halves of the dual store consistent.

## Key API

| Symbol | Role |
|---|---|
| `ShortTermMemory` | Thread-safe bounded FIFO buffer of LangChain messages for one session (`add`, `get_all`, `get_recent`, `clear`). |
| `ConversationHistory` | Append-only Markdown transcript per session with YAML front-matter and multi-channel tracking (`append`, `read`, `path`). |
| `LongTermMemory` | Cross-session fact store over SQLite + a `VectorStorePort` (`save`, `recall`, `expire`, `clear`); redacts secrets before persistence. |
| `MemoryEntry` | Pydantic model for one persisted fact: `id`, `session_id`, redacted `content`, `created_at`, `expires_at`. |
| `VectorStorePort` | Fase Z protocol the vector backend must satisfy; the notebook's `KeywordVectorStore` is a minimal conforming double. |

## Run it

```bash
uv run jupyter lab notebooks/memory_management.ipynb
uv run python examples/memory_management.py   # from the prismal repo
```

No API key required — the demo runs fully offline with injected fakes (temporary SQLite, in-memory vector store).

## Related

- [Supervisor quickstart](supervisor_quickstart.md) — the state machine these memories serve.
- [Composition root](composition_root.md) — how ports (vector store, checkpointer, audit) are assembled per tenant.
- [Kokoro deliberation](kokoro_deliberation.md) — another subsystem built on the same injection-first design.
