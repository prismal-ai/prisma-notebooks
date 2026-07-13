# Adaptive RAG

> **Notebook:** [`notebooks/rag/02_adaptive_rag.ipynb`](../../notebooks/rag/02_adaptive_rag.ipynb) · **Example:** [`examples/rag/02_adaptive_rag.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/02_adaptive_rag.py)

No single retrieval strategy wins everywhere. Keyword-heavy technical queries reward exact
lexical matching; abstract "why?" questions suffer a vocabulary gap that HyDE closes; vague
two-word queries benefit from multi-query fusion. Adaptive RAG puts a query classifier in
front of a set of specialized engines and routes each query to the one that fits its shape,
instead of forcing every query through the same pipeline.

Choose it over plain RAG when your traffic mixes query types and you want per-query strategy
selection without paying for an LLM router: the default classifier is deterministic regex —
zero extra LLM calls, fully reproducible — with an optional LLM classifier for edge cases.
The demo classifies ten NQ/TriviaQA-style queries across all six types and shows which
engine handled each.

## What it demonstrates

- Six-way query classification — `FACTUAL_SIMPLE`, `ABSTRACT`, `AMBIGUOUS`, `TECHNICAL`,
  `CONVERSATIONAL`, `MULTI_HOP` — via a deterministic regex classifier with confidence scores.
- Routing to four sub-engines (CRAG, HyDE, RAG-Fusion, Hybrid Search) that all share one
  ChromaDB collection.
- Classification accuracy measured over 10 labeled queries, plus a distribution of which
  engines actually ran.
- Graceful degradation: any query type whose preferred engine is not configured falls back
  to CRAG, so the facade always answers.
- The trade-offs between the regex classifier (free, deterministic) and the optional LLM
  classifier (`use_llm_classifier=True`).

## How it works

### One corpus, four engines

All sub-engines search the same `ChromaVectorStore`; Hybrid Search additionally needs a
BM25 index built over the raw corpus, since it fuses lexical and semantic scores:

```python
    # Shared vector store
    store = ChromaVectorStore(collection_name=collection_name)
    store.add_documents(BASE_DOCUMENTS)

    # Build sub-engines
    crag = CRAGPipeline(vector_store=store)
    hyde = HyDERetriever(vector_store=store)
    fusion = RAGFusionEngine(vector_store=store, n_queries=4)
    hybrid = HybridSearchEngine(vector_store=store, alpha=0.5)

    # Build BM25 index for Hybrid Search
    corpus = [doc.page_content for doc in BASE_DOCUMENTS]
    doc_ids = [doc.metadata["chunk_id"] for doc in BASE_DOCUMENTS]
    hybrid.build_index(corpus=corpus, doc_ids=doc_ids)
```

Only the CRAG pipeline is required by `AdaptiveRAGEngine`; the others are optional
injections. That is the degradation mechanism: a missing engine never errors a query type —
its queries simply route to CRAG.

### Classify by query shape

`classify_query()` runs a fixed cascade of precompiled regexes where *the first match
wins*, ordered so that the strongest structural signals dominate: multi-hop phrasing
("first X, then Y") is checked before conversational references ("what you mentioned
earlier"), then code-like tokens (`snake_case`, `API`, `foo.bar(`), then "why / how come",
then vagueness (≤ 2 words), then wh-factual words. Each rule carries a confidence encoding
how strong the heuristic is; an unmatched query defaults to `FACTUAL_SIMPLE` at 0.4. The
routing table the demo prints:

```python
    routing = [
        ("FACTUAL_SIMPLE", "CRAG", "direct facts, dates, names"),
        ("ABSTRACT       ", "HyDE", "questions 'why?', 'how does it work?'"),
        ("AMBIGUOUS      ", "RAG-Fusion", "short/vague queries (< 3 tokens)"),
        ("TECHNICAL      ", "Hybrid", "snake_case, CamelCase, API, SDK"),
        ("CONVERSATIONAL ", "CRAG", "references to previous context"),
        ("MULTI_HOP      ", "CRAG*", "'first X, then Y' structure"),
    ]
```

The mapping encodes *why each engine fits*: abstract questions have the query/document
vocabulary gap HyDE was built for; ambiguous queries need the query expansion that
RAG-Fusion provides; technical queries contain exact identifiers that lexical BM25 matches
better than embeddings. Multi-hop routes to CRAG as a fallback — its natural engine
(GraphRAG) is reserved for a later phase.

### Dispatch and evaluate

Every query goes through the same call; the result reports both the classification and the
engine that actually ran (they can differ when a preferred engine is unavailable):

```python
    for query_info in ADAPTIVE_QUERIES:
        result = await engine.search(query_info["query"], k=3)
        print_adaptive_result(query_info, result)

        if result.query_type == query_info["query_type"]:
            correct_classifications += 1

        engine_usage[result.strategy_used] = engine_usage.get(result.strategy_used, 0) + 1
```

Internally, async engines (CRAG, HyDE, Fusion) are awaited directly, while sync engines
(Hybrid, Hierarchical) are bridged through `asyncio.to_thread`, so `search()` stays `async`
regardless of the sub-engine's API shape. A `force_strategy=` argument bypasses the
classifier entirely when you already know which engine you want.

### Regex vs. LLM classification

The demo closes with the trade-off summary. With `use_llm_classifier=True` the engine asks
the LLM to pick one of the six categories (confidence 0.9 on success) and *falls back to
the regex classifier* on any LLM error or unrecognizable reply — so enabling it can only
add nuance, never a new failure mode:

```python
    print("  Regex (default):")
    print("    + No additional LLM cost")
    print("    + Deterministic and reproducible")
    print("    - May fail on ambiguous edge cases")
    print("  LLM (use_llm_classifier=True):")
    print("    + Better precision on edge cases")
    print("    + Understands natural language nuances")
    print("    - Extra cost of an LLM call")
    print("    - Falls back to regex if the LLM fails")
```

## Key API

| Symbol | Role |
|---|---|
| `AdaptiveRAGEngine(...)` | Facade over the sub-engines: a required CRAG pipeline plus optional HyDE / Fusion / Hybrid / Hierarchical engines and `use_llm_classifier` |
| `AdaptiveRAGEngine.search(query, k=5, force_strategy=None)` | Classifies, routes, and executes; returns an `AdaptiveResult` |
| `AdaptiveRAGEngine.classify_query(query)` | Deterministic regex classification → `(QueryType, confidence)` |
| `QueryType` | `StrEnum` of the six query categories |
| `AdaptiveResult` | `chunks`, `strategy_used` (engine that ran), `query_type`, `confidence` |
| `CRAGPipeline` / `HyDERetriever` / `RAGFusionEngine` / `HybridSearchEngine` | The injectable sub-engines |
| `ChromaVectorStore` | Shared vector store all engines search |

## Run it

```bash
uv run jupyter lab notebooks/rag/02_adaptive_rag.ipynb
uv run python examples/rag/02_adaptive_rag.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [CRAG — Corrective RAG](01_crag.md) — the required fallback engine every unmatched query routes to.
- [HyDE](04_hyde.md) — the engine Adaptive RAG selects for abstract queries.
- [Hybrid Search](05_hybrid_search.md) — the BM25 + semantic engine behind the TECHNICAL route.
- [RAG-Fusion](07_rag_fusion.md) — the multi-query engine behind the AMBIGUOUS route.
