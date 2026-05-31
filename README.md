# prisma-notebooks

> Executable Jupyter notebooks for the **[Prismal](https://github.com/prismal-ai/prismal)** framework — reasoning patterns, RAG engines, subgraphs, multimodal layer and extension surface, all with real datasets.

42 notebooks (`00_index.ipynb` + 41 examples) generated from the [`prismal/examples/`](https://github.com/prismal-ai/prismal/tree/main/examples) directory.

---

## Setup

```bash
git clone https://github.com/prismal-ai/prisma-notebooks
cd prisma-notebooks
uv venv && uv pip install -e .
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
uv run jupyter lab
```

Minimum: Python 3.13+, `uv`. If you don't use `uv`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
jupyter lab
```

---

## Structure

```
prisma-notebooks/
├── notebooks/
│   ├── 00_index.ipynb              ← navigable index
│   ├── patterns/    (9 notebooks)
│   ├── rag/         (9 notebooks)
│   ├── subgraphs/   (12 notebooks)
│   ├── multimodal/  (5 notebooks)
│   ├── extension/   (4 notebooks)
│   ├── multimodal_pipeline.ipynb
│   └── visualize_graphs.ipynb
├── data/                           ← real datasets
│   ├── arxiv/arxiv_papers.csv      (148 KB · 106 papers)
│   ├── github-issues/github_issues.csv  (2.1 MB · 60,001 issues)
│   └── medquad/medquad.csv         (22 MB · 60,049 medical QA)
├── README.md
├── pyproject.toml
├── LICENSE
└── .gitignore
```

---

## Suggested execution order — from simplest to most complex

This is the **recommended learning path**. Each level assumes concepts from the previous one. **41 notebooks across 9 levels**, ordered by increasing complexity.

### Level 1 · Framework basics (5 notebooks · 🔑 no API keys)

Start here if you've **never used Prismal**. Decorators, fluent builder, visualization — everything fits on a single page.

1. [`extension/custom_node.ipynb`](notebooks/extension/custom_node.ipynb) — the minimal `@prismal_node` decorator.
2. [`extension/custom_subgraph.ipynb`](notebooks/extension/custom_subgraph.ipynb) — fluent builder over `StateGraph`.
3. [`visualize_graphs.ipynb`](notebooks/visualize_graphs.ipynb) — generate Mermaid diagrams of any subgraph.
4. [`extension/discover_plugins_demo.ipynb`](notebooks/extension/discover_plugins_demo.ipynb) — plugin discovery via entry points.
5. [`extension/langchain_migration.ipynb`](notebooks/extension/langchain_migration.ipynb) — wrap an LCEL `Runnable` as a prismal node.

### Level 2 · Simple patterns (4 notebooks)

Reasoning patterns implemented as pure functions — easy to follow the flow.

6. [`patterns/09_parallel_dispatcher.ipynb`](notebooks/patterns/09_parallel_dispatcher.ipynb) — fan-out with `asyncio.gather` (🔑 no API).
7. [`patterns/06_reflection_loop.ipynb`](notebooks/patterns/06_reflection_loop.ipynb) — generate → critique → refine.
8. [`patterns/02_debate.ipynb`](notebooks/patterns/02_debate.ipynb) — N agents debate + Jaccard consensus.
9. [`patterns/08_swarm.ipynb`](notebooks/patterns/08_swarm.ipynb) — decentralized handoff between agents.

### Level 3 · Multimodal with mocks (6 notebooks · 🔑 no API keys)

How vision/audio/video agents work **without** calling real Whisper / VLM / FFmpeg — everything with injected callables.

10. [`multimodal/04_modality_router.ipynb`](notebooks/multimodal/04_modality_router.ipynb) — deterministic modality classification.
11. [`multimodal/05_multimodal_fusion.ipynb`](notebooks/multimodal/05_multimodal_fusion.ipynb) — `concat` / `moderator` / `moa` strategies.
12. [`multimodal/01_vision_agent.ipynb`](notebooks/multimodal/01_vision_agent.ipynb) — `VisionAgent` over arXiv figures.
13. [`multimodal/02_audio_agent.ipynb`](notebooks/multimodal/02_audio_agent.ipynb) — `AudioAgent` STT → reason → TTS.
14. [`multimodal/03_video_agent.ipynb`](notebooks/multimodal/03_video_agent.ipynb) — `VideoAgent` frames + audio + fusion.
15. [`multimodal_pipeline.ipynb`](notebooks/multimodal_pipeline.ipynb) — end-to-end pipeline (router → agents → fusion → formatter).

### Level 4 · RAG fundamentals (3 notebooks)

Start with the most production-used RAG, then see two complementary variants.

16. [`rag/01_crag.ipynb`](notebooks/rag/01_crag.ipynb) — Corrective RAG (baseline).
17. [`rag/05_hybrid_search.ipynb`](notebooks/rag/05_hybrid_search.ipynb) — BM25 lexical + semantic.
18. [`rag/04_hyde.ipynb`](notebooks/rag/04_hyde.ipynb) — Hypothetical Document Embeddings.

### Level 5 · Advanced patterns (5 notebooks)

Patterns requiring more state, more LLM calls, or structures such as trees / DAGs.

19. [`patterns/07_constitutional_ai.ipynb`](notebooks/patterns/07_constitutional_ai.ipynb) — review by principles.
20. [`patterns/01_tree_of_thoughts.ipynb`](notebooks/patterns/01_tree_of_thoughts.ipynb) — reasoning tree search.
21. [`patterns/05_mixture_of_agents.ipynb`](notebooks/patterns/05_mixture_of_agents.ipynb) — N proposers + aggregator.
22. [`patterns/04_llm_compiler.ipynb`](notebooks/patterns/04_llm_compiler.ipynb) — parallel DAG + Kahn.
23. [`patterns/03_lats.ipynb`](notebooks/patterns/03_lats.ipynb) — MCTS over actions (UCB1).

### Level 6 · Advanced RAG (6 notebooks)

Retrieval strategies that combine or rewrite queries, add hierarchies, or filter by modality.

24. [`rag/06_hierarchical_rag.ipynb`](notebooks/rag/06_hierarchical_rag.ipynb) — parent/child chunks.
25. [`rag/07_rag_fusion.ipynb`](notebooks/rag/07_rag_fusion.ipynb) — Reciprocal Rank Fusion.
26. [`rag/03_self_rag.ipynb`](notebooks/rag/03_self_rag.ipynb) — the LLM decides whether to retrieve.
27. [`rag/08_multi_vector_rag.ipynb`](notebooks/rag/08_multi_vector_rag.ipynb) — chunk + summary + N questions.
28. [`rag/02_adaptive_rag.ipynb`](notebooks/rag/02_adaptive_rag.ipynb) — facade that routes by query type.
29. [`rag/09_multimodal_rag.ipynb`](notebooks/rag/09_multimodal_rag.ipynb) — text + image + audio + video corpus with modality filtering.

### Level 7 · Simple subgraphs (5 notebooks)

Linear 4–6 node pipelines — easy to follow the flow.

30. [`subgraphs/04_code_review.ipynb`](notebooks/subgraphs/04_code_review.ipynb) — lint + sec + logic + suggester.
31. [`subgraphs/07_document_generation.ipynb`](notebooks/subgraphs/07_document_generation.ipynb) — plan → research → write → edit → format.
32. [`subgraphs/05_data_etl.ipynb`](notebooks/subgraphs/05_data_etl.ipynb) — extract → validate → transform → load + audit.
33. [`subgraphs/08_debate_consensus.ipynb`](notebooks/subgraphs/08_debate_consensus.ipynb) — proponent / opponent / moderator.
34. [`subgraphs/06_customer_service.ipynb`](notebooks/subgraphs/06_customer_service.ipynb) — classifier + FAQ + escalation.

### Level 8 · Advanced subgraphs (4 notebooks)

Pipelines with quality gates, parallel execution, and `interrupt()` for HITL.

35. [`subgraphs/01_ml_pipeline.ipynb`](notebooks/subgraphs/01_ml_pipeline.ipynb) — end-to-end ML with quality gate.
36. [`subgraphs/02_financial_analyst.ipynb`](notebooks/subgraphs/02_financial_analyst.ipynb) — technical + fundamental + risk analysis.
37. [`subgraphs/03_dev_pipeline.ipynb`](notebooks/subgraphs/03_dev_pipeline.ipynb) — PO → arch → dev → tests → QA → review.
38. [`subgraphs/09_hitl_approval.ipynb`](notebooks/subgraphs/09_hitl_approval.ipynb) — `interrupt()` + human decision.

### Level 9 · Hierarchical orchestrators (3 notebooks · most complex)

LLM-router that delegates to a domain of agents; combine everything above.

39. [`subgraphs/12_research_orchestrator.ipynb`](notebooks/subgraphs/12_research_orchestrator.ipynb) — `researcher` | `rag_agent`.
40. [`subgraphs/11_engineering_orchestrator.ipynb`](notebooks/subgraphs/11_engineering_orchestrator.ipynb) — `coder` | `codeact` | `planner` | `file_manager` | `skill_manager`.
41. [`subgraphs/10_analysis_orchestrator.ipynb`](notebooks/subgraphs/10_analysis_orchestrator.ipynb) — `data_analyst` | `ml_pipeline` | `dev_pipeline` | `financial_analyst` (4 demo modes).

---

## Included datasets

| Path | Rows | Size | Used by |
|------|------:|-------:|-----------|
| `data/arxiv/arxiv_papers.csv` | 106 | 148 KB | `multimodal/01_vision_agent`, `rag/09_multimodal_rag`, `subgraphs/12_research_orchestrator` |
| `data/github-issues/github_issues.csv` | 60,001 | 2.1 MB | `subgraphs/11_engineering_orchestrator` |
| `data/medquad/medquad.csv` | 60,049 | 22 MB | `rag/09_multimodal_rag`, `subgraphs/12_research_orchestrator` |

The remaining notebooks **embed** their own dataset (hard-coded samples for reproducibility).

The 4 notebooks that read these CSVs include a cell at the start (`_find_data_root()`) that automatically locates the `data/` folder, regardless of which subdirectory Jupyter is launched from.

---

## Environment variables

Create `.env` at the root:

```bash
ANTHROPIC_API_KEY=sk-ant-...   # primary, recommended
OPENAI_API_KEY=sk-...          # optional (Whisper, OpenAI fallback)
PRISMAL_HIERARCHICAL_MODE=true # enables orchestrators (Level 9)
```

**Notebooks without API keys (🔑):** all of Level 1 + 3 + part of Level 2; orchestrators run in simulation mode without network access.

**Notebooks with API keys (🔐):** all of RAG, MoA, ToT, Debate (real LLM mode), HITL Approval and the advanced subgraphs.

---

## Per-notebook conventions

1. **Cell 0** — markdown with name, dataset, expected result, setup and original docstring.
2. **Cell 1** — installs/applies `nest_asyncio` so that `await main()` works in cells.
3. **Cells 2…N** — alternating markdown/code following the sections of the original `.py`.
4. **Final cell** — `# await main()` ready to uncomment.

The 4 notebooks that read local datasets have two extra cells after `nest_asyncio`:
- Markdown explaining that `data/` is searched from the cwd.
- Code with `_find_data_root()` that assigns `DATA_ROOT` to a `pathlib.Path`.

---

## Regenerate from the prismal repo

The notebooks were generated from `prismal/examples/`. To regenerate them (for example, if you modify an example):

```bash
# Clone the prismal repo next to this one
cd ..
git clone https://github.com/prismal-ai/prismal
cd prisma-notebooks

# Copy the generators and run
cp -r ../prismal/notebooks/_generator ./
python _generator/build_notebooks.py
python _generator/build_index.py
python _generator/migrate_to_new_repo.py   # re-sync this repo
```

The generator parses the module docstring and section comments, producing notebooks with alternating markdown/code cells.

---

## Resources

- Prismal framework: <https://github.com/prismal-ai/prismal>
- Documentation: <https://prismal.dev/docs> *(planned)*
- PyPI: <https://pypi.org/project/prismal>
- Source examples: <https://github.com/prismal-ai/prismal/tree/main/examples>
- Hands-on guide with datasets (Obsidian): `Documentation/Prismal/Prismal-Datasets-Usage-Guide.md`

---

## License

MIT — see [LICENSE](LICENSE).
