# Hierarchical RAG — Parent/Child Chunking

> **Notebook:** [`notebooks/rag/06_hierarchical_rag.ipynb`](../../notebooks/rag/06_hierarchical_rag.ipynb) · **Example:** [`examples/rag/06_hierarchical_rag.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/06_hierarchical_rag.py)

Every plain RAG system faces a chunk-size dilemma: small chunks embed cleanly and match queries precisely, but hand the LLM a fragment with no surrounding context; large chunks give the LLM plenty of context, but their embeddings average many topics together and match queries poorly. Hierarchical (parent/child) RAG — `HierarchicalRAGEngine` in `prismal.rag.hierarchical` (SPEC-RAG-005) — refuses the trade-off by using **two chunk sizes at once**: it *searches* over small child chunks (~100 chars) and *returns* their enclosing parent chunks (~500 chars) to the LLM.

Choose it over plain RAG for long, structured documents — contracts, manuals, papers — where the phrase that matches a query ("shall not exceed the amounts paid…") is meaningless without the paragraph around it. The demo indexes CUAD-style commercial contracts (license, NDA, MSA), whose natural section → paragraph → clause structure is exactly what parent/child chunking exploits.

## What it demonstrates

- Two-level indexing: parent blocks of ~500 chars split into overlapping child chunks of ~100 chars, with **only the children embedded**.
- The `parent_id` + `parent_content` metadata trick: each child carries its full parent text, so no second store or lookup table is needed.
- Search precision from child granularity, generation context from parent granularity — measured by keyword hits on 5 CUAD-style clause questions.
- Deduplicated parents: multiple child hits from the same paragraph collapse into one parent, ranked by the best child score.
- Idempotent re-indexing: `index_document()` deletes prior chunks for the same source before inserting.

## How it works

### Indexing: split twice, embed once

`index_document(path)` loads the file through `DocumentProcessorFactory`, then chunks it twice. The document is first cut into non-overlapping parent blocks (`parent_chunk_size=500`); each parent is then cut into child chunks (`child_chunk_size=100`, `child_overlap=20`). Every child becomes a `Document` whose metadata records `source`, its own `chunk_id`, a generated `parent_id`, and — crucially — the complete `parent_content`. Only these child documents are added to the vector store, so the parent text rides along inside child metadata instead of a separate store. The method returns `(n_parents, n_children)` and calls `delete_by_source()` first, so re-indexing the same file never duplicates chunks. The demo indexes three contracts this way:

```python
    engine = HierarchicalRAGEngine(
        vector_store=ChromaVectorStore(collection_name="cuad_hierarchical"),
    )

    # Create temporary files and index them
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for contract in CUAD_CONTRACTS:
            contract_path = tmpdir_path / contract["filename"]
            contract_path.write_text(contract["content"], encoding="utf-8")
            await asyncio.to_thread(engine.index_document, contract_path)
```

Why does the small embedding matter? A 100-char child like "LICENSOR'S TOTAL CUMULATIVE LIABILITY SHALL NOT EXCEED THE AMOUNTS PAID" is almost a pure signal for a liability question. A 500-char embedding of the whole clause — mixing liability, damages taxonomy, and notice periods — sits farther from any single query in vector space. Constructor validation enforces the invariants (`child_chunk_size < parent_chunk_size`, `child_overlap < child_chunk_size`).

### Retrieval: match children, return parents

`search(query, k)` runs similarity search over the child index and returns a `HierarchicalSearchResult` whose `chunks` are the matched **child** hits — each with `content` (the ~100-char child text) and `relevance_score`. The full **parent** text rides along in `chunk.metadata["parent_content"]`, so a single search hit exposes both the precise child match and its surrounding parent context without a second lookup.

The demo prints the asymmetry this creates — tiny matched fragment, full returned context:

```python
    print("\n[Hierarchical indexing architecture]")
    print("  Full document")
    print("  └── Parent chunk (500 chars) ← used to GENERATE the answer")
    print("       └── Child chunk (100 chars) ← indexed in ChromaDB for SEARCH")
    print()
    print("  Advantage: child chunk = search precision")
    print("             parent chunk = rich context for the LLM")
```

### Evaluation: clause questions with expected keywords

Each question targets a specific clause and a keyword that only appears in the correct parent block — e.g. the NDA's confidentiality survival period ("five years") or the MSA's uptime guarantee ("99.9%"):

```python
    {
        "id": "CQ1",
        "question": "What is the duration of the confidentiality period in the NDA?",
        "relevant_contract": "nda_agreement.txt",
        "expected_keyword": "five",
        "clause_type": "confidentiality_term",
    },
    {
        "id": "CQ3",
        "question": "What SLA does CloudServices guarantee and what credits does it offer for breach?",
        "relevant_contract": "service_agreement.txt",
        "expected_keyword": "99.9",
        "clause_type": "service_level_agreement",
    },
```

The scoring loop simply checks whether the expected keyword appears in the returned text — a keyword like "SOC 2" may not be in the ~100-char child that matched "security measures", but it *is* in the ~500-char parent, which is exactly the point:

```python
    for question in CUAD_QUESTIONS:
        result = await asyncio.to_thread(engine.search, question["question"], 3)
        print_hierarchical_result(question, result)
```

The closing summary spells out why this suits legal documents: precision (small chunks match specific legal terms), context (the parent paragraph is what makes a clause interpretable), no truncation of critical context at generation time, and scalability to 100+ page contracts.

### Choosing the two sizes

The defaults (500/100 with an overlap of 20) are deliberately extreme so the demo makes the mechanism visible; in production the same ratios apply at larger scales (e.g. 2000/400). The trade-off to keep in mind when tuning:

```python
    print("\n[Comparison of chunk sizes]")
    print("  Child chunk : ~100 chars → high search precision")
    print("  Parent chunk: ~500 chars → sufficient context for generation")
    print(
        f"  Document    : {max(len(c['content']) for c in CUAD_CONTRACTS)} chars max → no indexing limit"
    )
```

Shrinking children sharpens matching but multiplies the number of embeddings (cost and index size grow); growing parents enriches context but eats into the LLM's context budget per retrieved hit — with `k=3` parents of 500 chars you spend ~1,500 chars of prompt, and that figure scales linearly with both knobs. The child overlap exists so that a phrase straddling a chunk boundary still appears intact in at least one child; without it, boundary-splitting a key term would make it invisible to search.

## Key API

| Symbol | Role |
|---|---|
| `HierarchicalRAGEngine(vector_store, parent_chunk_size=500, child_chunk_size=100, child_overlap=20, settings=None)` | The parent/child engine; validates that children are smaller than parents and overlap smaller than children. |
| `HierarchicalRAGEngine.index_document(path)` | Loads, double-chunks, and embeds child chunks (with `parent_id`/`parent_content` metadata); dedupes by source first; returns `(n_parents, n_children)`. |
| `HierarchicalRAGEngine.search(query, k=5)` | Searches the child index and returns the top-k matching child chunks, each carrying its parent's text in metadata. |
| `HierarchicalSearchResult` | Dataclass with `chunks` — matched child `RetrievedChunk`s (`content`, `relevance_score`, `metadata["parent_content"]`). |
| `ChromaVectorStore` | Default `VectorStorePort` adapter holding the child-chunk index. |

## Run it

```bash
uv run jupyter lab notebooks/rag/06_hierarchical_rag.ipynb
uv run python examples/rag/06_hierarchical_rag.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`.

## Related

- [Multi-Vector RAG](08_multi_vector_rag.md) — the other "index small, return big" design: chunks plus summaries plus hypothetical questions per document.
- [Hybrid Search — BM25 + Semantic](05_hybrid_search.md) — improves the *matching* signal; combine with hierarchical chunking, which improves the *returned* context.
- [Self-RAG](03_self_rag.md) — adds self-assessment on top of retrieval, useful when even parent context may be irrelevant.
- [CRAG](01_crag.md) — corrective retrieval that grades and filters what the retriever returns.
