# RAG-Fusion — Multi-Query + Reciprocal Rank Fusion

> **Notebook:** [`notebooks/rag/07_rag_fusion.ipynb`](../../notebooks/rag/07_rag_fusion.ipynb) · **Example:** [`examples/rag/07_rag_fusion.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/07_rag_fusion.py)

Plain RAG bets everything on a single phrasing: if the user's wording happens to sit far from the relevant document in embedding space, retrieval fails and no downstream cleverness can recover. RAG-Fusion — `RAGFusionEngine` in `prismal.rag.fusion` (SPEC-RAG-002) — hedges that bet. It asks an LLM for N-1 rephrasings of the query, runs one similarity search per phrasing in parallel, and merges the N ranked lists with **Reciprocal Rank Fusion (RRF)**: a rank-based voting scheme in which documents that surface across multiple rankings rise to the top.

Choose it over plain RAG when recall is the bottleneck — ambiguous or underspecified queries, corpora whose vocabulary diverges from how users talk, or multi-domain collections. The trade-off is explicit: one extra LLM call plus N searches per question. The demo uses a BEIR-style corpus (TREC-COVID, ArguAna, HotpotQA, FEVER) because BEIR is the standard benchmark where multi-query fusion shows its recall gains.

## What it demonstrates

- LLM-driven query expansion: `n_queries=4` means the original plus 3 generated paraphrases, all searched concurrently.
- The RRF formula `score(d) = Σ_{q in queries} 1/(k + rank_q(d))` with `k=60` (Cormack et al. 2009), worked through a hand-computed table.
- Why fusing **ranks** instead of scores is robust when per-query similarity scores are not comparable.
- Full result transparency: `FusionResult` exposes the generated queries, per-query top-k lists, and the fused ranking side by side.
- Recall@3 measured over 4 BEIR-style queries with known expected sources.

## How it works

### The RRF formula, by hand

Before touching the engine, the demo computes RRF for four documents across three hypothetical query rankings:

```python
    k = 60
    scenarios = [
        ("doc_A", 1, 2, 1),  # ranks high in all → winner
        ("doc_B", 3, 1, 5),  # appears in some
        ("doc_C", 10, 15, 2),  # only strong in Q3
        ("doc_D", None, 3, None),  # only in Q2
    ]

    for doc, r1, r2, r3 in scenarios:
        rrf = 0.0
        if r1:
            rrf += 1 / (k + r1)
        if r2:
            rrf += 1 / (k + r2)
        if r3:
            rrf += 1 / (k + r3)
```

Each ranked list a document appears in contributes `1/(k + rank)` — nothing else. The mechanics worth internalizing: a document consistently near the top of *several* lists (`doc_A`) beats one that is #1 in a single list, because contributions *sum* across queries; the constant `k=60` flattens the difference between rank 1 (`1/61`) and rank 3 (`1/63`), so one lucky first place cannot dominate broad agreement; and absence simply contributes zero rather than a penalty.

This is why RRF needs no calibration. Raw similarity scores from different query variants live on incomparable scales — one paraphrase may top out at 0.9 cosine similarity, another at 0.6, and averaging them would silently favor whichever variant happens to produce inflated scores. Ranks are scale-free: position 1 means the same thing in every list. RRF discards the scores entirely and keeps only the ordering, which makes it robust across heterogeneous queries (and even across heterogeneous retrievers). Contrast this with [Hybrid Search](05_hybrid_search.md), which fuses raw scores and therefore must normalize them and tune an α weight.

The library function behaves identically — chunks are identified by `(source, chunk_id)` and deduplicated, with every occurrence still contributing its rank:

```python
    sample_lists = [
        [RetrievedChunk("s1", "0", 0.9, "text1"), RetrievedChunk("s2", "1", 0.8, "text2")],
        [RetrievedChunk("s2", "1", 0.7, "text2"), RetrievedChunk("s1", "0", 0.6, "text1")],
    ]
    fused = reciprocal_rank_fusion(sample_lists, k=60)
```

### Engine setup

The engine wraps a vector store and a query-variant count. Internally it resolves an LLM via `ProviderRegistry` and builds the rephrasing prompt through `SecurePromptBuilder` (the user query is untrusted input and never f-stringed into a template):

```python
    store.add_documents(docs)

    return RAGFusionEngine(
        vector_store=store,
        n_queries=n_queries,  # number of query variations to generate
    )
```

`search(query, k)` then performs three steps: (1) one LLM call parses `n_queries - 1` numbered rephrasings out of the model output (a short list is tolerated; an LLM failure raises `FusionError`); (2) `asyncio.gather` dispatches one similarity search per phrasing concurrently, so latency is roughly one search plus one LLM call, not N sequential searches; (3) `reciprocal_rank_fusion` merges the ranked lists, and the fused chunks carry their RRF score as `relevance_score`.

### Reading a fusion result

`FusionResult` keeps every intermediate artifact, which the demo prints so you can watch recall being manufactured — a source missed by the original phrasing but found by a paraphrase accumulates RRF score and enters the fused top-3:

```python
    print(f"\n  Queries generated by RAG-Fusion ({len(result.queries)}):")
    for i, q in enumerate(result.queries, 1):
        print(f"    {i}. {q}")

    print("\n  Per-query results:")
    for i, (_query, chunks) in enumerate(
        zip(result.queries, result.per_query_results, strict=False)
    ):
        top_sources = [c.source for c in chunks[:3]]
        print(f"    Q{i + 1}: top-3 → {top_sources}")

    print("\n  Fused ranking (RRF score):")
    for chunk in result.chunks[:5]:
        expected_mark = "→ " if chunk.source == query_info["expected_source"] else "  "
        print(f"    {expected_mark}[{chunk.relevance_score:.4f}] {chunk.source}")
```

The queries are chosen so the original wording does not literally match the target document — "advantages of universal basic income for fighting poverty" must reach an ArguAna passage phrased around "UBI", "welfare", and "poverty traps". The summary is honest about the cost side too: higher recall and robustness to user wording, no weighting parameters to calibrate — but N+1 calls and more latency than standard RAG.

## Key API

| Symbol | Role |
|---|---|
| `RAGFusionEngine(vector_store, n_queries=4, rrf_k=60, settings=None)` | Multi-query engine; `n_queries` counts the original, so the default generates 3 LLM variants. |
| `RAGFusionEngine.search(query, k=5)` | Async: generate variants → N parallel searches (top-k each) → RRF fusion; raises `FusionError` if variant generation fails. |
| `reciprocal_rank_fusion(ranked_lists, k=60)` | Standalone RRF: dedupes by `(source, chunk_id)` and returns chunks ordered by descending RRF score. |
| `FusionResult` | Dataclass: fused `chunks`, the `queries` actually searched, and `per_query_results` for debugging. |
| `RetrievedChunk` | Shared result type; after fusion its `relevance_score` holds the RRF score. |
| `ChromaVectorStore` | Default `VectorStorePort` adapter used for each per-variant similarity search. |

## Run it

```bash
uv run jupyter lab notebooks/rag/07_rag_fusion.ipynb
uv run python examples/rag/07_rag_fusion.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [Hybrid Search — BM25 + Semantic](05_hybrid_search.md) — score-based fusion of two retrievers over one query; RRF is the rank-based alternative over many queries.
- [HyDE](04_hyde.md) — the other query-transformation strategy: one hypothetical answer instead of N paraphrases.
- [Adaptive RAG](02_adaptive_rag.md) — routes each query to the engine that fits it, RAG-Fusion included.
- [CRAG](01_crag.md) — complements fusion's recall boost with a corrective relevance check on what comes back.
