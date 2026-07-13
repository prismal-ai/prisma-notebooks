# Multimodal RAG — Text, Image, Audio, Video

> **Notebook:** [`notebooks/rag/09_multimodal_rag.ipynb`](../../notebooks/rag/09_multimodal_rag.ipynb) · **Example:** [`examples/rag/09_multimodal_rag.py`](https://github.com/prismal-ai/prismal/blob/main/examples/rag/09_multimodal_rag.py)

Plain RAG assumes your knowledge lives in text. Real corpora do not: papers carry
figures, support archives carry voice recordings, training material carries video.
Multimodal RAG (`prismal.rag.multimodal`, SPEC-MM-RAG-001) indexes all four modalities
into a **single collection**, tagging every item with `modality` and `source_uri`
metadata, so one query can retrieve a paragraph, a figure caption, and a call transcript
side by side — and can also be *restricted* to just the modalities you want.

The key design decision (DD-MM-003) is the fallback path: when no cross-modal embedder
(such as CLIP) is installed, the engine embeds the **textual proxy** of each media item —
captions for images, transcripts for audio and video — and logs an explicit
`multimodal_rag_textual_fallback` warning. Choose this architecture when your corpus
mixes media types; choose plain RAG when everything is already text.

## What it demonstrates

- Building a mixed corpus — arXiv abstracts and MedQuAD Q&A (text), figure captions
  (image), ATIS-style utterance transcripts (audio), scene descriptions (video) — every
  item tagged with `metadata["modality"]` and `metadata["source_uri"]`.
- Captions and transcripts acting as the textual proxy for non-text media, so retrieval
  works without CLIP or any cross-modal extra installed.
- `search(query, k)` with no filter: results interleave modalities, ranked purely by
  relevance.
- `search(query, modalities=[Modality.IMAGE])` and multi-modality filters
  (`[TEXT, VIDEO]`): the same collection, sliced by media type at query time.
- That `MultimodalRAGEngine` depends only on the `VectorStorePort` shape — the demo
  swaps in a tiny in-memory keyword store and everything still works.

## How it works

### The engine only needs a `VectorStorePort`

`MultimodalRAGEngine` types its store against the port, not against Chroma. The notebook
exploits that to stay dependency-free, substituting a keyword-overlap store:

```python
# `MultimodalRAGEngine` only needs:
#   - .add_documents(list[Document]) -> list[str]
#   - .similarity_search(query, k) -> list[(Document, score)]
class InMemoryStore:
    """Minimal mock of `ChromaVectorStore` using overlapping-keyword score."""

    def __init__(self) -> None:
        self._docs: list[Document] = []

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids: list[str] = []
        for i, d in enumerate(documents):
            self._docs.append(d)
            ids.append(f"doc-{len(self._docs):04d}")
        return ids
```

In production you would let the engine index real files via `engine.index(path)`, which
sniffs each file with `MediaValidator`, routes it to the matching `ImageLoader` /
`AudioLoader` / `VideoLoader`, and returns per-modality chunk counts.

### Modality metadata on every indexed item

Each `Document` carries two contract fields: `modality` (one of `text`, `image`,
`audio`, `video`) and `source_uri` (where the original media lives — the vector store
never holds the media bytes themselves). For images, the page content is the caption:

```python
def _image_docs() -> list[Document]:
    """Figure captions (no cross-modal embedding → textual fallback)."""
    return [
        Document(
            page_content="Figure: architecture diagram of a transformer encoder block "
                         "with multi-head attention and feed-forward layers.",
            metadata={"modality": "image", "source_uri": "arxiv://2604.02185#fig1",
                      "source": "arxiv-2604.02185", "type": "diagram"},
        ),
        Document(
            page_content="Figure: a histopathology slide showing glaucoma-induced optic "
                         "nerve atrophy in a 65-year-old patient.",
            metadata={"modality": "image", "source_uri": "medquad://glaucoma/fig",
                      "source": "medquad-glaucoma", "type": "medical-photo"},
        ),
    ]
```

Audio items follow the same pattern with their transcript as content
(`"Show me all flights from Boston to Denver on Tuesday morning."` tagged
`modality="audio"`), and video items with a scene description. Because the searchable
text *is* the caption or transcript, a plain text embedding (or here, keyword overlap)
retrieves media items exactly as it retrieves text.

### The textual fallback is explicit, not silent

Constructing the engine without a `cross_modal_embedder` is legal and logged:

```python
    # 2. Instantiate engine (no cross_modal_embedder → textual fallback)
    engine = MultimodalRAGEngine(vector_store=store)  # type: ignore[arg-type]
```

The engine emits `logger.warning("multimodal_rag_textual_fallback", ...)` at
construction, so an operator can see at a glance whether retrieval is cross-modal (CLIP
embeddings via the `[multimodal-embed]` extra) or caption-based.

### Modality filtering in `search()`

`search()` accepts an optional `modalities` allowlist. Internally it **over-fetches**
`k * 4` hits when a filter is set — post-filtering discards non-matching modalities, so
the engine needs headroom to still return `k` results — then maps each hit's metadata
string to the canonical `Modality` enum and stops at `k`:

```python
    # 4. Modality-filtered search
    for q in ["transformer architecture", "glaucoma diagnosis"]:
        chunks = await engine.search(q, k=3, modalities=[Modality.IMAGE])
        for c in chunks:
            assert c.modality is Modality.IMAGE
            print(f"    [IMAGE] {c.source_uri} score={c.score:.2f}")

    # 5. Multi-modality search (text + video)
    chunks = await engine.search("cooking", k=4, modalities=[Modality.TEXT, Modality.VIDEO])
```

Every result is a frozen `MultimodalRetrievedChunk` carrying `modality`, `source_uri`,
`score`, `content`, and the full metadata dict — enough for a caller to fetch the
original media, render a citation, or route by media type downstream.

## Key API

| Symbol | Role |
|---|---|
| `MultimodalRAGEngine(vector_store, *, cross_modal_embedder=None, ...)` | Engine over any `VectorStorePort`; warns `multimodal_rag_textual_fallback` when no cross-modal embedder is injected |
| `MultimodalRAGEngine.index(path)` | Indexes a file or directory, sniffing media kind via `MediaValidator` and routing to image/audio/video loaders; returns chunk counts per `Modality` |
| `MultimodalRAGEngine.search(query, *, k=5, modalities=None)` | Async search; over-fetches `k * 4` when filtering, post-filters by modality, returns top-k chunks |
| `MultimodalRetrievedChunk` | Frozen result: `chunk_id`, `content`, `modality`, `source_uri`, `score`, `metadata` |
| `Modality` | Canonical enum (`TEXT` / `IMAGE` / `AUDIO` / `VIDEO`) from `prismal.agents.multimodal`; metadata strings like `"video_frame"` map onto it |

## Run it

```bash
uv run jupyter lab notebooks/rag/09_multimodal_rag.ipynb
uv run python examples/rag/09_multimodal_rag.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` for the full
pipeline; the notebook's in-memory store keeps the retrieval demo itself lightweight.

## Related

- [Multi-Vector RAG](08_multi_vector_rag.md) — also indexes derived textual
  representations (summaries, questions) pointing back to a source, much like captions
  point back to media.
- [Hybrid Search](05_hybrid_search.md) — keyword + semantic scoring, relevant when your
  textual proxies are short caption-like strings.
- [Federated RAG](10_federated_rag.md) — the complementary scaling axis: one query over
  many collections instead of many modalities in one collection.
