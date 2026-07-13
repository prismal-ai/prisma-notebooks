# Hybrid Search — BM25 + Semantic

> **Notebook:** [`notebooks/rag/05_hybrid_search.ipynb`](../../notebooks/rag/05_hybrid_search.ipynb) · **Example:** [`examples/rag/05_hybrid_search.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/05_hybrid_search.py)

Hybrid search combines two retrieval signals that fail in opposite ways: **BM25**, a sparse lexical scorer that rewards exact term matches, and **dense semantic search**, which embeds queries and documents into a shared vector space where paraphrases land close together. Pure semantic retrieval routinely misses queries built from proper nouns, version numbers, or rare technical tokens (the embedding smooths them away), while pure lexical retrieval misses queries that describe a concept without using the document's vocabulary. `HybridSearchEngine` (`prismal.rag.hybrid`, SPEC-RAG-003) fuses both with a single tunable weight.

Choose hybrid search over plain RAG whenever your corpus mixes precise identifiers with prose — news, legal text, source code, product catalogs — or when you cannot predict whether users will search by exact keyword or by concept. The demo uses an AG News-style corpus (World, Sports, Business, Science/Tech) precisely because news articles contain both signal types.

## What it demonstrates

- Linear score fusion: `score(d) = α·semantic(d) + (1-α)·bm25_norm(d)`, with `α` configurable per engine and per call.
- An α sweep (`0.0 / 0.3 / 0.5 / 0.7 / 1.0`) over the same corpus, showing how the balance shifts retrieval outcomes.
- Query classes where **BM25 wins** (exact proper nouns like "NVIDIA H100 Jensen Huang", version strings like "ChromaDB 0.5").
- Query classes where **semantic wins** (paraphrases like "government regulation of intelligent systems" → the EU AI Act article).
- Per-query BM25 max-normalization so raw Okapi scores become comparable with cosine-style semantic scores in `[0, 1]`.

## How it works

### Setup: two indexes over one corpus

The engine wraps a vector store (Chroma by default, via the `VectorStorePort` contract) for the dense side and an in-memory `BM25Okapi` index for the sparse side. Both are built over the same documents; the BM25 index must be built explicitly with a corpus and a parallel `doc_ids` list (conventionally `"{source}:{chunk_id}"`, which is how fused candidates are matched back to semantic hits):

```python
    store = ChromaVectorStore(collection_name=f"ag_news_hybrid_{alpha:.1f}")

    store.add_documents(docs)

    engine = HybridSearchEngine(
        vector_store=store,
        alpha=alpha,
    )

    # Build BM25 index with the same corpus
    corpus = [f"{art['title']}. {art['text']}" for art in AG_NEWS_ARTICLES]
    doc_ids = [art["id"] for art in AG_NEWS_ARTICLES]
    engine.build_index(corpus=corpus, doc_ids=doc_ids)
```

If `search()` is called before `build_index()`, the engine degrades gracefully to semantic-only retrieval (effectively `α = 1.0`) rather than failing. Note the scale caveat: `BM25Okapi` lives in memory, so beyond ~100K documents you would swap in an inverted-index backend (Tantivy, OpenSearch).

### Scoring: normalize, then fuse

BM25 produces unbounded, query-dependent scores, so they cannot be averaged with semantic similarities directly. The engine max-normalizes BM25 scores to `[0, 1]` *per query*, then takes the **union** of both candidate sets — a document found only by BM25 still enters the fused ranking (with semantic score 0), and vice versa. Each candidate gets the weighted sum, and the top-k by fused score is returned as `RetrievedChunk` objects whose `relevance_score` is the fused value.

### Queries designed to separate the two signals

The demo corpus is paired with queries labeled by which retriever should dominate:

```python
    # BM25 wins: exact keywords, proper nouns, technical terms
    {
        "id": "HQ1",
        "query": "NVIDIA H100 GPU data center revenue Jensen Huang",
        "type": "keyword_exact",
        "expected_source": "ANG004",
        "reason": "Exact proper nouns → BM25 beats semantic",
    },
    # Semantic wins: synonyms, paraphrases, concepts
    {
        "id": "HQ3",
        "query": "Government regulation of intelligent systems in Europe",
        "type": "semantic_paraphrase",
        "expected_source": "ANG006",
        "reason": "Paraphrase of 'EU AI Act' — semantic captures the concept",
    },
```

The intuition: "H100" and "Jensen Huang" are rare tokens with high inverse document frequency — BM25's TF-IDF core makes them nearly deterministic matches, while an embedding may dilute them among generically "GPU-related" articles. Conversely, "intelligent systems regulation" shares almost no vocabulary with "EU AI Act", so only the dense space bridges them.

### The α sweep

Each query is re-run at five α values against a fresh engine, checking whether the expected document lands in the top-3:

```python
async def run_alpha_comparison(query_info: dict) -> dict[float, bool]:
    """Compare results across different α values for a query."""
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = {}

    for alpha in alphas:
        engine = await setup_hybrid(alpha)
        chunks = engine.search(query_info["query"], k=3)
        results[alpha] = evaluate_retrieval(query_info, chunks)

    return results
```

The demo closes with a practical selection guide:

```python
    guidelines = [
        (0.0, "BM25 only", "Legal documents, source code, exact search"),
        (0.3, "BM25 dominant", "Technical corpora with specialized vocabulary"),
        (0.5, "Balance (default)", "General use — best compromise"),
        (0.7, "Semantic dominant", "Conversational text, frequent synonyms"),
        (1.0, "Semantic only", "Concept search, cross-lingual retrieval"),
    ]
```

`α = 0.5` is a sound default; move toward 0 when your users search with identifiers and toward 1 when they search with descriptions. `search()` also accepts a per-call `alpha` override, so a single indexed engine can serve both query styles.

## Key API

| Symbol | Role |
|---|---|
| `HybridSearchEngine(vector_store, alpha=0.5, settings=None)` | Hybrid retriever; `alpha` is the semantic weight in `[0, 1]` (validated, `ValueError` otherwise). |
| `HybridSearchEngine.build_index(corpus, doc_ids)` | Builds the in-memory `BM25Okapi` index; `corpus` and `doc_ids` must be parallel lists. |
| `HybridSearchEngine.search(query, k=5, alpha=None)` | Returns up to *k* `RetrievedChunk`s ordered by fused score; optional per-call `alpha` override; semantic-only fallback when no BM25 index exists. |
| `HybridSearchEngine.has_index()` | Whether `build_index()` has been called successfully. |
| `ChromaVectorStore` | Default `VectorStorePort` adapter backing the semantic side (`prismal.rag.vector_store` re-exports it from `rag/stores/chroma.py`). |
| `RetrievedChunk` | Result dataclass (`source`, `chunk_id`, `relevance_score`, `content`) shared across prismal RAG engines. |

## Run it

```bash
uv run jupyter lab notebooks/rag/05_hybrid_search.ipynb
uv run python examples/rag/05_hybrid_search.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [RAG-Fusion — Multi-Query + Reciprocal Rank Fusion](07_rag_fusion.md) — fuses *rankings* instead of scores, so no normalization or α tuning is needed.
- [HyDE](04_hyde.md) — attacks the same vocabulary-mismatch problem from the semantic side, by embedding a hypothetical answer.
- [Hierarchical RAG — Parent/Child Chunking](06_hierarchical_rag.md) — improves what you *return* to the LLM rather than how you rank it.
- [Adaptive RAG](02_adaptive_rag.md) — a router that can pick hybrid search only for the query types that need it.
