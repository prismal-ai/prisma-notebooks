# Swapping the vector store backend to LanceDB by configuration (Fase Z)

> **Notebook:** [`notebooks/vector_store_lancedb.ipynb`](../notebooks/vector_store_lancedb.ipynb) · **Example:** [`examples/vector_store_lancedb.py`](https://github.com/prismal-ai/prismal/blob/main/examples/vector_store_lancedb.py)

Fase Z inverts vector search as a hexagonal port, mirroring what Fase Y did for tools. Consumers — the RAG engines (`engine`, `hyde`, `fusion`, `self_rag`, `hybrid`, `hierarchical`, `multi_vector`, `crag`) and the memory layer — depend only on the `VectorStorePort` Protocol and never construct a backend themselves. `VectorStoreFactory.create(settings, collection_name)` reads `settings.vector_store_backend` and builds the matching adapter from `prismal/rag/stores/`. Chroma remains the default, so existing installs see zero breakage; the alternatives are opt-in extras: `[lancedb]`, `[sqlite-vec]`, `[qdrant]`, `[pgvector]`.

This notebook performs the swap that motivates the design: it flips one settings field to `"lancedb"`, gets an **embedded** LanceDB store back from the factory (no server process, data lives on local disk), verifies the object structurally conforms to the port, indexes three documents, and runs a similarity search whose scores obey the port-wide contract. No consumer code changes — only configuration.

## What it demonstrates

- A configuration-only backend swap: `Settings(vector_store_backend="lancedb", vector_store_path=...)` is the entire change; every RAG/memory consumer works unchanged.
- The factory as the single construction point: `VectorStoreFactory.create(settings, collection_name)` returns a `VectorStorePort`, never a concrete class in consumer code.
- Graceful degradation when the extra is missing: backend SDK imports are deferred inside each adapter, so an absent `lancedb` package surfaces as a catchable `VectorStoreBackendUnavailable` pointing at `pip install 'prismal[lancedb]'` — not an `ImportError` at module import time.
- Structural conformance via `conforms_to(store, VectorStorePort)` — the port is a `@runtime_checkable` Protocol.
- The score contract (SPEC-VS-002): `similarity_search` returns `(Document, score)` tuples with `score ∈ [0, 1]`, higher = more relevant, regardless of the backend's native metric.

## How it works

### Imports: the port, the factory, and the unavailability error

```python
from langchain_core.documents import Document

from prismal.agents.extension import VectorStorePort, conforms_to
from prismal.core.config import Settings
from prismal.core.exceptions import VectorStoreBackendUnavailable
from prismal.rag.vector_store_factory import VectorStoreFactory
```

`VectorStorePort` lives in `agents/extension/ports.py` next to the other hexagonal ports and defines five members: a `collection_name` property, `add_documents(documents) -> list[str]`, `similarity_search(query, k=5) -> list[tuple[Document, float]]`, `delete_by_source(source)`, and `delete_collection()`. Conforming implementations include `ChromaVectorStore` (the default, moved from `rag/vector_store.py`, which stays as a re-export shim), `LanceDBVectorStore`, `SqliteVecVectorStore`, `QdrantVectorStore`, `PgVectorStore`, and the deterministic `FakeVectorStore` test double.

### Selecting the backend by settings

```python
# Configuration-only backend swap — no code change in any consumer.
settings = Settings(
    vector_store_backend="lancedb",
    vector_store_path="data/db/vectors",
)

try:
    store: VectorStorePort = VectorStoreFactory.create(settings, collection_name="demo")
except VectorStoreBackendUnavailable as exc:
    # The base install does not ship LanceDB; the error guides to the extra.
    print(exc)
    print("Install it with: pip install 'prismal[lancedb]'")
    return
```

`vector_store_backend` defaults to `"chroma"`; `vector_store_path` is the on-disk location for embedded backends (server backends like Qdrant-in-server-mode or pgvector use `vector_store_url` plus `vector_store_api_key`/`user`/`password` instead). The legacy `chroma_path` setting is kept as a backward-compatible alias, reconciled via `Settings.resolve_vector_store_path()`. LanceDB and sqlite-vec are both fully embedded — setting the backend to `"sqlite_vec"` here would exercise the in-process SQLite option, and `"chroma"` restores the default behaviour.

The `try/except` is not decorative: each adapter defers its backend SDK import to construction time (`import lancedb` happens inside `LanceDBVectorStore.__init__`), so the base install stays slim and a missing extra fails with a targeted, actionable exception rather than breaking imports for users who never asked for LanceDB.

The backend landscape the factory selects from, at a glance:

| Backend | Deployment | Extra | Native metric handling |
|---|---|---|---|
| `chroma` | Embedded (default) | none — base install | Cosine similarity, already `[0, 1]` (`identity`) |
| `lancedb` | Embedded | `[lancedb]` | Distance, lower = better (`from_distance`) |
| `sqlite_vec` | Embedded, in-process SQLite | `[sqlite-vec]` | Distance (`from_distance`) |
| `qdrant` | Embedded or server | `[qdrant]` | Cosine similarity (`identity`) |
| `pgvector` | Server (PostgreSQL) | `[pgvector]` | Cosine distance (`from_cosine_distance`) |

Embedded backends need only `vector_store_path`; the server ones read `vector_store_url` and credentials. An unknown backend string is rejected by the factory with a clear error, and each adapter also owns its own construction (embeddings come from `EmbeddingsFactory`, so the embedding model follows its own configuration axis independently of the store).

### Indexing and searching through the port

Once built, the store is used purely through the Protocol. A quick assertion confirms structural conformance, then three documents go in and a query comes back ranked:

```python
assert conforms_to(store, VectorStorePort)

store.add_documents(
    [
        Document(page_content="Prismal is an agent framework.", metadata={"source": "intro"}),
        Document(page_content="LanceDB is an embedded vector DB.", metadata={"source": "db"}),
        Document(page_content="Cosine similarity ranks relevance.", metadata={"source": "math"}),
    ]
)

for doc, score in store.similarity_search("What is Prismal?", k=3):
    print(f"{score:.3f}  {doc.metadata['source']:8}  {doc.page_content}")
```

The scores deserve attention. Backends disagree wildly on their native metrics — Chroma reports a cosine similarity where higher is better, LanceDB reports a *distance* where lower is better. SPEC-VS-002 resolves this at the port boundary: the port *defines* the contract (`score ∈ [0, 1]`, higher = more relevant) and each adapter *normalizes* its native metric to meet it. The shared normalizers live in `rag/stores/_normalize.py`: `identity` (clamped pass-through for already-conforming similarities), `from_distance` (maps a non-negative distance to `[0, 1]` via `1 / (1 + d)`, so `d = 0` scores `1.0` and decays monotonically), and `from_cosine_distance` (inverts `1 - similarity` and clamps). `LanceDBVectorStore` applies `from_distance` to every raw result — which is why the demo can format `score:.3f` and trust that the `intro` document ranks first with the highest score, whatever backend is configured.

### Lifecycle: the rest of the port

The demo exercises the read/write half of the contract; the remaining two methods cover cleanup. `delete_by_source(source)` removes every document whose `metadata["source"]` matches — the convention all prismal loaders stamp on ingestion — and is defined as best-effort, so re-indexing a refreshed document set is an idempotent delete-then-add. `delete_collection()` drops the whole collection/table. Both are part of the Protocol, so consumers can manage retention without knowing which backend is underneath. Adapter-level failures surface as `VectorStoreError` (which `ChromaStoreError` now subclasses), keeping exception handling backend-agnostic too.

Because every consumer types against `VectorStorePort` and builds through the factory, this same indexing/search flow is exactly what `rag/engine.py`, the seven advanced RAG engines, and `memory/long_term.py` execute internally — swap the backend under any of them and they neither know nor care. For tests, `FakeVectorStore` (in `rag/vector_store_factory.py`) implements the same port with pre-seeded `(Document, score)` tuples and zero I/O.

## Key API

| Symbol | Role |
|---|---|
| `VectorStorePort` | `@runtime_checkable` Protocol: `collection_name`, `add_documents`, `similarity_search`, `delete_by_source`, `delete_collection` |
| `VectorStoreFactory.create(settings, collection_name)` | Single construction point; maps `settings.vector_store_backend` to the concrete adapter |
| `Settings.vector_store_backend` | Backend selector — `"chroma"` (default), `"lancedb"`, `"sqlite_vec"`, `"qdrant"`, `"pgvector"` |
| `Settings.vector_store_path` | On-disk location for embedded backends (`chroma_path` kept as a legacy alias) |
| `VectorStoreBackendUnavailable` | Raised at construction when the backend's optional extra is not installed |
| `conforms_to(obj, port)` | Structural conformance check against a runtime-checkable Protocol |
| `similarity_search(query, k)` | Returns `(Document, score)` with `score ∈ [0, 1]`, higher = better (SPEC-VS-002) |
| `FakeVectorStore` | Deterministic, I/O-free port implementation for tests |

## Run it

```bash
uv run jupyter lab notebooks/vector_store_lancedb.ipynb
uv run python examples/vector_store_lancedb.py   # from the prismal repo
```

No API key is required — the demo runs offline. The notebook's final cell ships with the sync entry point commented out (`# main()`); uncomment and run it (no `await` needed). To actually exercise the LanceDB path you need the extra installed — `pip install 'prismal[lancedb]'` — otherwise the demo still runs but takes the `VectorStoreBackendUnavailable` branch and prints the install hint, which is itself half the lesson.

## Related

- [Runtime composition root (Phase R)](composition_root.md) — `build_runtime()` carries the vector store as one of the five composed ports, with per-tenant collections.
- [Host-style tool provider composition and injection (Fase Y)](tool_provider_host.md) — the mirror inversion this phase is modeled on.
- [Environment config source (Fase W)](config_source_env.md) — where the `Settings` values driving the factory come from.
- [Corrective RAG (CRAG)](rag/01_crag.md) — a RAG consumer that retrieves through this port.
