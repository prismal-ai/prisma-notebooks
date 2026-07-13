# Self-RAG

> **Notebook:** [`notebooks/rag/03_self_rag.ipynb`](../../notebooks/rag/03_self_rag.ipynb) · **Example:** [`examples/rag/03_self_rag.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/03_self_rag.py)

Plain RAG retrieves unconditionally: even "What is diabetes?" — a definition any LLM knows
from parametric memory — pays the latency of a vector search and risks having its answer
distracted by whatever chunks came back. Self-RAG hands control of the pipeline to the LLM
itself through *reflection tokens*: before answering, the model emits `RETRIEVE` or
`NO_RETRIEVE` to decide whether external context is needed at all; after answering, it
grades its own output with a support token (`SUPPORTED` / `PARTIALLY_SUPPORTED` /
`UNSUPPORTED`) and a utility score (1–5).

Choose Self-RAG over plain RAG when your query mix includes many general-knowledge
questions (skipping retrieval cuts latency and vector-store load), or when downstream
consumers need a machine-readable quality signal attached to every answer. The demo runs it
over PubMedQA-style biomedical abstracts, where some questions need a specific figure from
an abstract and others do not.

## What it demonstrates

- The `RETRIEVE` / `NO_RETRIEVE` decision: the LLM skips retrieval for questions it can
  answer from prior knowledge and retrieves for questions requiring specific data.
- Conditional delegation: when retrieval is chosen, the query runs through the full
  `CRAGPipeline` (grading, filtering, cited generation).
- Self-assessment after generation: a `SUPPORTED`-family token plus a 1–5 utility score for
  every answer, retrieved or not.
- Permissive, fail-safe token parsing: unparseable decisions default to the *safer* outcome
  (retrieve when unsure; `UNSUPPORTED` with utility 1 when assessment fails).
- Decision accuracy measured against per-question expectations (`expected_retrieve`) over
  six PubMedQA queries.

## How it works

### Index the abstracts

Five real-style PubMed abstracts (mRNA vaccines, metformin, tau protein, CRISPR CAR-T, the
gut-brain axis) are indexed with `pmid` and `source` metadata:

```python
    docs = [
        Document(
            page_content=abstract["abstract"],
            metadata={
                "source": abstract["source"],
                "pmid": abstract["pmid"],
                "title": abstract["title"],
                "chunk_id": str(i),
                "dataset": "PubMedQA",
            },
        )
        for i, abstract in enumerate(PUBMED_ABSTRACTS)
    ]

    store.add_documents(docs)
```

The questions are labeled with the retrieval decision a well-calibrated model *should*
make — "What was the efficacy of mRNA COVID-19 vaccines in the phase 3 trial?" needs the
95% figure from the abstract, while a plain definition does not:

```python
    {
        "id": "PQ1",
        "question": "What was the efficacy of mRNA COVID-19 vaccines in the phase 3 trial?",
        "expected_retrieve": True,
        "expected_source": "pubmed_covid_vaccines",
        "reason": "Requires a specific figure from the abstract (95%)",
    },
    {
        "id": "PQ2",
        "question": "What is diabetes?",
        "expected_retrieve": False,
        "expected_source": None,
        "reason": "General definition the LLM knows without retrieval",
    },
```

### The retrieval decision

The pipeline first asks the LLM for exactly one token — retrieve or not. This is the core
inversion of Self-RAG: retrieval becomes *on demand* rather than unconditional. Parsing is
deliberately permissive (models rarely emit control tokens perfectly), and when the reply
contains neither token the pipeline defaults to `RETRIEVE` and flags the result — grounding
an answer unnecessarily is cheaper than hallucinating one, so uncertainty resolves toward
retrieval. On the `RETRIEVE` path the query is delegated to `CRAGPipeline`, reusing its
relevance grading and cited generation; on `NO_RETRIEVE` the LLM answers directly from
general knowledge.

```python
    pipeline = SelfRAGPipeline(vector_store=store)

    # Self-RAG control tokens
    print("\n[Self-RAG reflection tokens]")
    print("  RETRIEVAL: RETRIEVE | NO_RETRIEVE")
    print("  SUPPORT  : SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED")
    print("  UTILITY  : 1 (low) → 5 (high)")
```

### Self-assessment of the answer

After generating, the pipeline shows the LLM its own answer next to the context it used
(or "(no context)") and asks for a verdict in the form
`<SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED> Utility:<1-5>`. The support token tells you
whether the answer is actually grounded in the retrieved chunks — an answer can cite
context and still drift from it. Failures here also degrade safely: an unparseable
assessment yields `(UNSUPPORTED, 1)`, a pessimistic default that lets downstream consumers
filter low-confidence outputs instead of trusting them.

```python
    for question in PUBMED_QUESTIONS:
        result = await pipeline.run(question["question"])
        print_self_rag_result(question, result)

        # Statistics
        retrieved = result.retrieval_decision.value == "RETRIEVE"
        if retrieved == question["expected_retrieve"]:
            retrieve_correct += 1
```

### Why it beats always-retrieve

The demo closes by summarizing what the reflection tokens buy you:

```python
    benefits = [
        ("Selectivity", "Does not retrieve when the LLM already knows the answer"),
        ("Self-evaluation", "The LLM grades the relevance of its own sources"),
        ("Adaptability", "Adjusts the answer based on the support level found"),
        ("Efficiency", "Fewer vector store calls = lower latency on simple queries"),
    ]
```

## Key API

| Symbol | Role |
|---|---|
| `SelfRAGPipeline(vector_store, crag_pipeline=None, settings=None)` | The pipeline; builds a default `CRAGPipeline` over the store when none is injected |
| `SelfRAGPipeline.run(query)` | Decide → (conditionally) retrieve via CRAG → generate → self-assess; returns a `SelfRAGResult` |
| `RetrievalDecision` | `StrEnum`: `RETRIEVE` \| `NO_RETRIEVE` |
| `SupportedDecision` | `StrEnum`: `SUPPORTED` \| `PARTIALLY_SUPPORTED` \| `UNSUPPORTED` |
| `SelfRAGResult` | `answer`, `retrieval_decision`, `support_decision`, `utility_score` (1–5), `chunks` (empty when no retrieval), `used_fallback` |
| `CRAGPipeline` | Retrieval + grading + generation delegate on the `RETRIEVE` path |
| `ChromaVectorStore` | Vector store backing the internal CRAG pipeline |

## Run it

```bash
uv run jupyter lab notebooks/rag/03_self_rag.ipynb
uv run python examples/rag/03_self_rag.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [CRAG — Corrective RAG](01_crag.md) — the pipeline Self-RAG delegates to whenever it decides to retrieve.
- [Adaptive RAG](02_adaptive_rag.md) — the complementary decision: not *whether* to retrieve, but *which engine* to use.
- [HyDE](04_hyde.md) — improves what retrieval returns, where Self-RAG decides whether retrieval happens at all.
