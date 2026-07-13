# CRAG — Corrective RAG

> **Notebook:** [`notebooks/rag/01_crag.ipynb`](../../notebooks/rag/01_crag.ipynb) · **Example:** [`examples/rag/01_crag.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/01_crag.py)

Plain RAG has a structural blind spot: a top-k similarity search *always* returns k chunks,
even when nothing in the knowledge base actually answers the question. Those weakly related
chunks flow straight into the prompt, and the model dutifully hallucinates an answer around
them. CRAG (Corrective RAG) inserts a correction stage between retrieval and generation: an
LLM grades every retrieved chunk against the query, chunks below a relevance threshold are
discarded, and when *nothing* survives, a web-search fallback fires instead of generating
from noise.

Choose CRAG over plain RAG when your knowledge base is small or incomplete, when queries
regularly fall outside it, or when you need per-chunk relevance scores and source citations
you can audit. The demo runs the pipeline over a SQuAD 2.0-style corpus of Wikipedia
contexts (Transformers, BERT, Python, RAG, LLMs).

## What it demonstrates

- The five-step CRAG pipeline — RETRIEVE → GRADE → FILTER → DECIDE → GENERATE — end to end
  over an embedded SQuAD 2.0 corpus.
- LLM relevance grading: every chunk gets a 0.0–1.0 score, with the raw vector-similarity
  score as a robust fallback when the LLM reply cannot be parsed.
- The 0.5 relevance threshold that removes weakly related context before generation.
- The web-search fallback (a stub in this phase) that guarantees an answer when no chunk
  passes the filter.
- Verifiable citations: `CRAGResult.sources` records exactly which graded chunks the answer
  was built from.

## How it works

### Index the corpus

The example wraps five Wikipedia contexts as LangChain `Document`s with `source`, `title`,
and `chunk_id` metadata, then indexes them in a `ChromaVectorStore` collection:

```python
from prismal.rag.crag import CRAGPipeline, CRAGResult
from prismal.rag.vector_store import ChromaVectorStore
```

```python
    store = ChromaVectorStore(collection_name=collection_name)

    # Create LangChain documents with metadata
    docs = [
        Document(
            page_content=ctx["content"],
            metadata={
                "source": ctx["source"],
                "title": ctx["title"],
                "chunk_id": str(i),
                "dataset": "squad_2.0",
            },
        )
        for i, ctx in enumerate(SQUAD_CONTEXTS)
    ]

    store.add_documents(docs)
```

The metadata matters: CRAG surfaces `source` and `chunk_id` on every graded chunk, so the
final answer can cite where each piece of context came from.

### Run the five-step pipeline

The pipeline itself needs only the store — the LLM is resolved from Prismal's provider
registry, and each `run()` call executes all five steps:

```python
    # Create CRAG pipeline
    pipeline = CRAGPipeline(vector_store=store)
    print("  ✓ CRAG pipeline initialized")
    print("  Relevance threshold: 0.5 (chunks with score < 0.5 are discarded)")
```

```python
    steps = [
        ("1. RETRIEVE", "ChromaDB similarity_search(query, k=5)"),
        ("2. GRADE   ", "LLM scores each chunk 0.0-1.0"),
        ("3. FILTER  ", "Keep chunks with score >= 0.5"),
        ("4. DECIDE  ", "If all filtered out → web fallback"),
        ("5. GENERATE", "LLM answers with context + citations"),
    ]
```

The key idea is that grading is a *second opinion, decoupled from embedding geometry*.
Cosine similarity measures closeness in vector space, not answerability — a chunk about
BERT is "close" to a question about GPT without answering it. The LLM grader reads the
chunk against the query and can say "related, but does not answer this". If the grader's
reply is not a parseable float, the pipeline quietly falls back to the similarity score
rather than failing, so one flaky model response never breaks a query.

### Inspect the graded sources

Each result exposes the per-chunk scores, which the demo renders as bars:

```python
    for chunk in result.sources:
        score_bar = "█" * int(chunk.relevance_score * 10) + "░" * (
            10 - int(chunk.relevance_score * 10)
        )
        print(
            f"    [{score_bar}] {chunk.relevance_score:.2f} — "
            f"{chunk.source} (chunk {chunk.chunk_id})"
        )

    if result.used_web_fallback:
        print("  ⚠ Web fallback triggered (no chunk passed the 0.5 threshold)")
```

### Force the web fallback

The final section asks a question the corpus cannot possibly answer. Every retrieved chunk
grades below 0.5, the filter empties the context, and the DECIDE step injects the fallback
chunk (`source="web_fallback"`) so the pipeline still produces an answer instead of an
empty prompt:

```python
    off_topic_result = await pipeline.run(
        "What are the best Michelin-starred sushi restaurants in Tokyo?"
    )
    print(f"  Fallback triggered: {off_topic_result.used_web_fallback}")
```

In this phase the fallback is a stub (full web-search integration is deferred), but the
control flow — *never generate from context that failed grading* — is the real CRAG
contract, and `used_web_fallback` makes the corrective path observable downstream.

## Key API

| Symbol | Role |
|---|---|
| `CRAGPipeline(vector_store, settings=None)` | Corrective pipeline; wires the configured LLM via the provider registry |
| `CRAGPipeline.run(query)` | Executes RETRIEVE → GRADE → FILTER → DECIDE → GENERATE; returns a `CRAGResult` |
| `CRAGResult` | `answer` (cited LLM answer), `sources` (surviving chunks), `used_web_fallback` flag |
| `RetrievedChunk` | Graded chunk: `source`, `chunk_id`, `relevance_score` in [0, 1], `content` |
| `ChromaVectorStore(collection_name)` | Chroma wrapper; `add_documents(docs)` and `similarity_search(query, k)` returning `(Document, score)` pairs |

## Run it

```bash
uv run jupyter lab notebooks/rag/01_crag.ipynb
uv run python examples/rag/01_crag.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [Adaptive RAG](02_adaptive_rag.md) — routes queries between engines and uses CRAG as its universal fallback.
- [Self-RAG](03_self_rag.md) — lets the LLM decide *whether* to retrieve at all, then delegates retrieval to CRAG.
- [HyDE](04_hyde.md) — attacks the complementary problem: retrieval missing the right chunks for abstract queries.
