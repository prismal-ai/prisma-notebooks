# Federated RAG — Multi-Node Search with Fusion and Re-Ranking

> **Notebook:** [`notebooks/rag/10_federated_rag.ipynb`](../../notebooks/rag/10_federated_rag.ipynb) · **Example:** [`examples/rag/10_federated_rag.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/10_federated_rag.py)

In many organizations knowledge cannot live in one vector store: the engineering team
indexes its runbooks on its own node, research indexes papers on another, and no single
collection is allowed (or able) to hold everything. Federated RAG
(`prismal.rag.federated`, SPEC-021 / T-213) answers a query against **all** of them at
once: the local `RAGEngine` is searched synchronously, every remote node is queried in
parallel over HTTP, and the partial rankings are fused — deduplicated by `chunk_id` and
re-ranked by relevance score — into one answer.

Choose it over plain RAG when knowledge is distributed across nodes by ownership,
geography, or access boundaries. Its defining property is **failure isolation**: a dead
node is logged and skipped, so the query degrades to partial results instead of failing.
This demo runs 100% offline — the HTTP layer is overridden with in-memory corpora, while
the fan-out, fault tolerance, and fusion code paths are the real production code.

## What it demonstrates

- Fan-out of a single query to one local engine plus N remote nodes via
  `asyncio.gather`, each node returning its own top-k.
- Fusion with `_merge_and_rerank`: deduplication by `chunk_id` (the highest-scoring
  replica wins) followed by a descending sort on `relevance_score`.
- Failure isolation: an offline node raises `ConnectionError` inside the gather, gets
  logged and skipped, and the query still returns results from the healthy nodes.
- A deliberately replicated chunk (`kb-rag-100` on two nodes) proving that replicas
  collapse to a single result carrying the best score.
- Recall@3 over three domain-targeted queries against a deterministic keyword scorer —
  no LLM, no embeddings, no API key.

## How it works

### One corpus per node, plus a dead node

The synthetic knowledge base spreads nine documents across a local node (general
knowledge), an engineering node, a research node — and an archived node that is marked
offline to exercise the fault-tolerance path. One chunk is intentionally replicated:

```python
REMOTE_NODES: list[dict[str, Any]] = [
    {
        "name": "nodo-ingenieria",
        "url": "https://ingenieria.example.internal",
        "corpus": [
            {
                "chunk_id": "ing-latencia-010",
                "text": (
                    "Para optimizar la latencia de inferencia del modelo se "
                    "recomienda cuantización INT8, batching dinámico y caché "
                    "de prefijos en el servidor de inferencia."
                ),
            },
            # ... observability and CI chunks
        ],
    },
    # ... "nodo-investigacion" (holds a replica of kb-rag-100)
    {
        # Nodo caído: demuestra que la federación devuelve resultados
        # parciales en lugar de fallar (SPEC-021 criterio 3).
        "name": "nodo-archivado",
        "url": "https://archivado.example.internal",
        "corpus": [],
        "offline": True,
    },
]
```

### Deterministic scoring, so the demo needs no model

Each node scores documents by token overlap — `score = |query ∩ doc| / |query|`, bounded
to `[0, 1]` — which makes every run reproducible and keeps the focus on the *federation*
mechanics rather than embedding quality:

```python
def _keyword_search(
    corpus: list[dict[str, str]],
    node_name: str,
    query: str,
    k: int,
) -> list[RetrievedChunk]:
    """Búsqueda determinista: score = |query ∩ doc| / |query| en [0, 1]."""
    q_tokens = _tokens(query)
    if not q_tokens:
        return []

    scored: list[RetrievedChunk] = []
    for doc in corpus:
        overlap = len(q_tokens & _tokens(doc["text"]))
        score = round(overlap / len(q_tokens), 4)
        if score > 0.0:
            scored.append(
                RetrievedChunk(
                    source=node_name,
                    chunk_id=doc["chunk_id"],
                    relevance_score=score,
                    content=doc["text"],
                )
            )

    scored.sort(key=lambda c: c.relevance_score, reverse=True)
    return scored[:k]
```

### Overriding only the transport, keeping the production flow

In production, each remote node exposes `GET /api/v1/rag/search` with JWT bearer auth
(SPEC-018 RBAC), and `FederatedRAGEngine._query_remote_node` calls it with `httpx`. The
demo subclasses the engine and replaces **only** that method — the parallel gather, the
per-node exception handling, and `_merge_and_rerank` all run unmodified:

```python
class OfflineFederatedRAGEngine(FederatedRAGEngine):
    """FederatedRAGEngine con los nodos remotos servidos desde memoria."""

    async def _query_remote_node(
        self,
        node: dict[str, Any],
        query: str,
        k: int,
    ) -> list[RetrievedChunk]:
        """Simula GET /api/v1/rag/search contra el corpus del nodo."""
        if node.get("offline"):
            msg = f"nodo '{node['name']}' no responde (connection refused)"
            raise ConnectionError(msg)
        return _keyword_search(node["corpus"], node["name"], query, k)
```

Inside the real `search()`, remote tasks are launched with
`asyncio.gather(*remote_tasks, return_exceptions=True)`: any node that raises comes back
as an exception object, is logged as `federated_rag_node_failed`, and simply contributes
nothing. That is the failure-isolation guarantee — one refused connection never poisons
the fused ranking. Total latency is bounded by the slowest healthy node (the engine's
`timeout_seconds` caps the wait per remote request).

### Fusion: dedupe by `chunk_id`, keep the best score, sort descending

`_merge_and_rerank` is small enough to demo in isolation. Given two replicas of `doc-1`
scored 0.62 and 0.91 on different nodes, the fused ranking keeps exactly one — the 0.91
version — and orders everything by score:

```python
    chunks = [
        RetrievedChunk("nodo-a", "doc-1", 0.62, "réplica en nodo-a"),
        RetrievedChunk("nodo-b", "doc-1", 0.91, "réplica en nodo-b"),
        RetrievedChunk("local", "doc-2", 0.75, "chunk único local"),
        RetrievedChunk("nodo-a", "doc-3", 0.40, "chunk único remoto"),
    ]

    merged = _merge_and_rerank(chunks)
```

The main run then executes three federated queries (deployment approval → local node,
inference latency → engineering node, federated retrieval → the replicated chunk, where
either node may win) and reports recall@3. The archived node fails on every query, yet
every query succeeds — partial results, no exception.

## Key API

| Symbol | Role |
|---|---|
| `FederatedRAGEngine(local_engine=None, remote_nodes=None, timeout_seconds=15)` | Composes a local `RAGEngine` with remote node dicts (`name`, `url`); defaults load RAG-capable nodes from `config/network_nodes.yaml` |
| `FederatedRAGEngine.search(query, k=5)` | Async: local search + parallel remote queries (`asyncio.gather(..., return_exceptions=True)`), failed nodes skipped, fused result returned |
| `FederatedRAGEngine._query_remote_node(node, query, k)` | The HTTP+JWT transport seam (`GET /api/v1/rag/search`); the demo overrides exactly this method to serve corpora from memory |
| `_merge_and_rerank(chunks)` | Pure fusion step: dedupe by `chunk_id` keeping the highest `relevance_score`, then sort descending |
| `RetrievedChunk` | Shared result type (`source`, `chunk_id`, `relevance_score`, `content`) from `prismal.rag.crag` |

## Run it

```bash
uv run jupyter lab notebooks/rag/10_federated_rag.ipynb
uv run python examples/rag/10_federated_rag.py   # from the prismal repo
```

No API key required — the demo runs fully offline with injected in-memory fakes.

## Related

- [RAG-Fusion](07_rag_fusion.md) — fuses rankings from multiple *queries* against one
  store; federated RAG fuses rankings from one query against multiple *stores*.
- [Hybrid Search](05_hybrid_search.md) — merges BM25 and semantic rankings, another
  instance of combining partial rankings into one list.
- [Multimodal RAG](09_multimodal_rag.md) — the complementary scaling axis: many
  modalities in one collection instead of one query over many nodes.
