# HyDE — Hypothetical Document Embeddings

> **Notebook:** [`notebooks/rag/04_hyde.ipynb`](../../notebooks/rag/04_hyde.ipynb) · **Example:** [`examples/rag/04_hyde.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/04_hyde.py)

Queries and documents occupy different regions of embedding space. A question like "Why are
transformers better than RNNs for NLP?" is short, interrogative, and phrased in question
vocabulary; the passage that answers it is long, declarative, and full of technical terms
the question never mentions. Embedding the query directly and searching for nearby
documents therefore suffers a *vocabulary mismatch* — the geometric gap between how people
ask and how documents explain. HyDE closes that gap by asking the LLM to first draft a
hypothetical document that would answer the query, then embedding *that draft* and
searching with it instead.

The trick works because retrieval only needs the right neighborhood, not the right facts: a
fake answer written in document style lands geometrically near the real answers even when
its details are wrong. Choose HyDE over plain RAG for abstract and conceptual queries
("why?", "how does it work?"); skip it when queries already contain the corpus's exact
keywords, since it adds one LLM call of latency for little gain. The demo compares HyDE
against direct-query CRAG retrieval over MS MARCO-style passages.

## What it demonstrates

- The HyDE flow: query → LLM generates a hypothetical document → the hypothesis (not the
  query) drives the vector search.
- A head-to-head comparison on the same corpus: HyDE vs. direct-query retrieval (CRAG),
  over 3 abstract and 2 concrete queries with known expected sources.
- Customizing the hypothesis generator via `hypothesis_prompt` to match the corpus's
  register (academic/technical English).
- Inspecting the generated hypothesis and its embedding through `HyDEResult`.
- Practical selection criteria: when the extra LLM call pays for itself and when it does not.

## How it works

### Index the passages and customize the hypothesis prompt

Six MS MARCO-style passages (attention, gradient descent, overfitting, transfer learning,
embeddings, RAG) share one Chroma collection between HyDE and the CRAG baseline. The
hypothesis prompt is tuned so generated drafts read like the corpus — an excerpt from a
technical article, not a chat reply:

```python
    # Custom prompt to generate hypothetical documents in English
    hypothesis_prompt = (
        "You are an expert in artificial intelligence and machine learning. "
        "Write a concise technical paragraph (3-5 sentences) that directly "
        "answers the following question, as if it were an excerpt from "
        "an academic article or technical documentation: {query}"
    )

    hyde_retriever = HyDERetriever(
        vector_store=store,
        hypothesis_prompt=hypothesis_prompt,
    )

    crag_pipeline = CRAGPipeline(vector_store=store)
```

Matching the corpus register is the whole point: the closer the hypothesis's vocabulary and
style are to the real documents, the closer its embedding lands to them.

### Search with the hypothesis, not the query

The retriever generates the hypothesis, embeds it (the vector is exposed on the result for
inspection), and runs the similarity search using the hypothesis text. The demo prints the
two flows side by side:

```python
    print("  Standard RAG:")
    print("    Query → embed(query) → similarity_search → chunks")
    print()
    print("  HyDE:")
    print("    Query → LLM generates hypothetical document H")
    print("         → embed(H) → similarity_search → chunks")
    print("  ")
    print("  The key: embed(H) ≈ embed(real_document) >> embed(query)")
```

That last line is the geometry argument in one inequality: the hypothetical document is a
much better proxy for the target passage than the raw query, because document-to-document
similarity is what the embedding space actually measures well.

### Compare HyDE against direct-query retrieval

Each query runs through both retrievers, and the demo checks which one surfaced the
expected source in its top 3:

```python
    # Comparison of retrieved chunks
    hyde_sources = [c.source for c in hyde_result.chunks[:3]]
    crag_sources = [c.source for c in crag_chunks[:3]]

    print("\n  Retrieved chunks:")
    print(f"    HyDE (hypothetical): {hyde_sources}")
    print(f"    CRAG (direct query): {crag_sources}")
```

The queries are deliberately split: abstract ones ("Why are transformers better than
RNNs?") have almost no lexical overlap with the passages that answer them — HyDE's home
turf — while concrete ones ("gradient descent optimizer Adam learning rate") already speak
the corpus's language, so direct embedding works fine and HyDE's extra LLM call buys
nothing.

### When to use it

```python
    recommendations = [
        ("✓ Use HyDE  ", "Abstract questions: 'why?', 'how does it work?'"),
        ("✓ Use HyDE  ", "Query vocabulary very different from the corpus"),
        ("✓ Use HyDE  ", "Technical domain with specialized jargon"),
        ("✗ Avoid HyDE", "Queries with exact keywords from the corpus"),
        ("✗ Avoid HyDE", "Critical latency (HyDE adds 1 LLM call)"),
        ("✗ Avoid HyDE", "Limited token budget"),
    ]
```

A caveat worth internalizing: HyDE inherits the LLM's blind spots. If the model knows
nothing about the topic, the hypothesis may land in the wrong neighborhood entirely —
which is why [Adaptive RAG](02_adaptive_rag.md) reserves HyDE for the query shapes where it
statistically wins, instead of using it everywhere.

## Key API

| Symbol | Role |
|---|---|
| `HyDERetriever(vector_store, settings=None, hypothesis_prompt=None)` | Retriever that searches with a generated hypothesis; prompt overrides the built-in default |
| `HyDERetriever.retrieve(query, k=5)` | Generate hypothesis → embed it → similarity search; returns a `HyDEResult` |
| `HyDEResult` | `chunks` (retrieved via the hypothesis), `hypothesis` (the generated text), `hypothesis_embedding` (its vector, for inspection) |
| `RetrievedChunk` | Per-chunk `source`, `chunk_id`, `relevance_score`, `content` (shared with CRAG) |
| `CRAGPipeline` | Direct-query baseline used for the comparison |
| `ChromaVectorStore` | Shared store; `similarity_search(text, k)` returns `(Document, score)` pairs |

## Run it

```bash
uv run jupyter lab notebooks/rag/04_hyde.ipynb
uv run python examples/rag/04_hyde.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [Adaptive RAG](02_adaptive_rag.md) — routes ABSTRACT queries to HyDE automatically.
- [CRAG — Corrective RAG](01_crag.md) — the direct-query baseline HyDE is compared against.
- [RAG-Fusion](07_rag_fusion.md) — the other query-transformation technique: many query variants instead of one hypothesis.
- [Multi-Vector RAG](08_multi_vector_rag.md) — the mirror image: indexes hypothetical *questions* per document at ingest time.
