# Multi-Vector RAG — Chunks, Summaries, and Hypothetical Questions

> **Notebook:** [`notebooks/rag/08_multi_vector_rag.ipynb`](../../notebooks/rag/08_multi_vector_rag.ipynb) · **Example:** [`examples/rag/08_multi_vector_rag.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/08_multi_vector_rag.py)

Plain RAG embeds each document chunk exactly once, so a query only matches when its
phrasing happens to land near the document's own vocabulary in embedding space. That
assumption breaks constantly with technical corpora: a user asks *"How do multimodal LLMs
handle visual perception?"* while the paper says *"vision encoders that process image
patches as tokens"*. Multi-Vector RAG (`prismal.rag.multi_vector`) attacks this mismatch
at indexing time by embedding every chunk under **three representations** — the raw chunk,
an LLM-written summary, and N LLM-generated hypothetical questions — all tied back to the
same parent document through a shared `doc_id`.

Choose it over plain RAG when your corpus uses specialized vocabulary (scientific papers,
API docs, legal text) and users phrase queries in their own words. The trade-off is
explicit: roughly 3× the indexing cost (LLM calls plus extra vectors) in exchange for
three independent "access angles" into each document.

## What it demonstrates

- Indexing 5 arXiv-style ML/AI abstracts, each expanded into chunk + summary + 3
  hypothetical questions — all vectors sharing a parent `doc_id`.
- Why question embeddings close the query/document phrasing gap: a question generated
  from a chunk is *shaped like* the queries users actually type.
- Search-time deduplication by `doc_id`: the best-scoring representation wins, but the
  result always carries the original chunk content.
- The `matched_representations` diagnostic — which surface (chunk / summary / question)
  produced each hit, useful for auditing what the extra index actually buys you.
- Recall@3 over 5 queries deliberately engineered to match via different representations.

## How it works

### A corpus where vocabulary diverges from queries

The dataset is a set of real arXiv abstracts (KOSMOS-1, Llama 2, Self-RAG, Mistral 7B,
LLM embeddings). Each is dense with technical terms that a casual query would never use:

```python
ARXIV_PAPERS = [
    {
        "arxiv_id": "2307.09288",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "filename": "llama2_paper.txt",
        "content": (
            "In this work, we develop and release Llama 2, a collection of pretrained and "
            "fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion "
            # ...
            "We used Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy "
            "Optimization (PPO) and rejection sampling to align the models with human preferences. "
        ),
    },
    # ... 4 more papers
]
```

### Indexing: one `doc_id`, three embedding surfaces

The engine takes any `VectorStorePort` implementation (here the default
`ChromaVectorStore`) and a per-chunk question budget:

```python
async def setup_multivector_rag() -> MultiVectorRAGEngine:
    engine = MultiVectorRAGEngine(
        vector_store=ChromaVectorStore(collection_name="arxiv_multivector"),
        n_questions=3,  # generate 3 hypothetical questions per chunk
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for paper in ARXIV_PAPERS:
            paper_path = tmpdir_path / paper["filename"]
            full_content = f"Title: {paper['title']}\n\nAbstract:\n{paper['content']}"
            paper_path.write_text(full_content, encoding="utf-8")
            await engine.index_document(paper_path)
```

Inside `index_document()`, every loaded chunk gets a fresh `doc_id` and then up to
`1 + 1 + n_questions` vector documents: the chunk itself (always), a 2–3 sentence LLM
summary, and N numbered hypothetical questions the chunk could answer. Summary and
question generation are **best-effort** — if an LLM call fails, the raw chunk
representation is still indexed, so no document is ever lost. Both generation prompts go
through `SecurePromptBuilder`, never f-strings, because chunk content is untrusted input.

### Queries engineered to hit each representation

Each demo query targets one representation on purpose, so the run shows *which* surface
earns its keep:

```python
MULTIVECTOR_QUERIES = [
    {
        "id": "MVQ1",
        "query": "How do multimodal LLMs handle visual perception?",
        "expected_source_prefix": "kosmos1",
        "match_type": "summary",
        "description": "Match via summary (conceptual) — the text uses different vocabulary",
    },
    {
        "id": "MVQ3",
        "query": "reflection tokens RETRIEVE IsSUP IsUSE special tokens self-reflection",
        "expected_source_prefix": "self_rag",
        "match_type": "chunk",
        "description": "Match via direct chunk — exact technical keywords",
    },
    # ... question-style queries for llama2 and mistral, summary for embeddings
]
```

The intuition: keyword-heavy queries hit the **chunk** index directly; conceptual
paraphrases land on the **summary**; natural-language questions embed closest to the
**question** representations, because those were generated to look exactly like questions.

### Search: over-fetch, dedupe by `doc_id`, return the parent chunk

`search()` over-fetches `k * 4` raw hits so that after collapsing multiple
representations of the same document it can still return `k` unique docs. Per `doc_id`
the best-scoring representation sets the ranking score, but the caller sees which
surfaces matched:

```python
    result = await engine.search(query_info["query"], k=5)

    # Show which representations matched
    if result.matched_representations:
        print("\n  Representations that matched:")
        for doc_id, representations in list(result.matched_representations.items())[:3]:
            print(f"    {doc_id}: {representations}")
```

The demo closes with recall@3 across the 5 queries, a per-representation match-type
tally, and the honest cost comparison: Multi-Vector buys higher recall and precision at
3× the indexing cost of standard single-chunk RAG.

## Key API

| Symbol | Role |
|---|---|
| `MultiVectorRAGEngine(vector_store, n_questions=3, settings=None)` | Engine that indexes each chunk under chunk / summary / question representations; LLM resolved lazily via `ProviderRegistry` |
| `MultiVectorRAGEngine.index_document(path)` | Loads and chunks a document, generates summaries and hypothetical questions best-effort, writes all vectors; returns `(n_docs, n_vectors)` |
| `MultiVectorRAGEngine.search(query, k=5)` | Over-fetches `k * 4` hits, deduplicates by `doc_id` keeping the best score, returns a `MultiVectorResult` |
| `MultiVectorResult` | `.chunks` (one `RetrievedChunk` per unique doc, score-ordered) + `.matched_representations` (`{doc_id: ["question", ...]}`) |
| `RetrievedChunk` | Result item: `source`, `chunk_id`, `relevance_score`, `content` (reused from `prismal.rag.crag`) |
| `ChromaVectorStore` | Default `VectorStorePort` adapter backing the three-representation index |

## Run it

```bash
uv run jupyter lab notebooks/rag/08_multi_vector_rag.ipynb
uv run python examples/rag/08_multi_vector_rag.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — summaries
and hypothetical questions are generated by a real model at indexing time.

## Related

- [HyDE](04_hyde.md) — the mirror-image trick: generate a hypothetical *document* from
  the query, instead of hypothetical *questions* from the document.
- [Hierarchical RAG](06_hierarchical_rag.md) — another parent/child index where matches
  on small units resolve back to a larger parent context.
- [RAG-Fusion](07_rag_fusion.md) — attacks the same phrasing gap at query time with
  multi-query rewriting and rank fusion.
