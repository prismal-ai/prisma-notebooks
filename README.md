# prisma-notebooks

> Executable Jupyter notebooks for the **[Prismal](https://github.com/prismal-ai/prismal)** framework — reasoning patterns, RAG engines, subgraphs, multimodal layer and extension surface, all with real datasets.

74 notebooks (`00_index.ipynb` + 73 examples) generated from the [`prismal/examples/`](https://github.com/prismal-ai/prismal/tree/main/examples) directory.

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
│   ├── rag/         (10 notebooks)
│   ├── subgraphs/   (12 notebooks)
│   ├── multimodal/  (5 notebooks)
│   ├── extension/   (4 notebooks)
│   ├── multimodal_pipeline.ipynb
│   ├── visualize_graphs.ipynb
│   ├── supervisor & multi-agent (4):      supervisor_quickstart, hierarchical_supervisor,
│   │                                      parallel_research_fanout, subagent_spawner
│   ├── specialized agents (5):            meta_learner, skill_creator_agent, memory_management,
│   │                                      multimodal_ingestion, blind_review_pipeline
│   ├── opt-in phases (9):                 kokoro_deliberation, skynet_{swarm,specialist_swarm,direct_api},
│   │                                      budget_{governance,graph_integration},
│   │                                      a2a_{server,remote_node,tool_provider}
│   └── infrastructure & hardening (13):   composition_root, config_source_{env,custom},
│                                          tool_provider_{host,custom}, vector_store_lancedb,
│                                          node_typesafety, observability_integration,
│                                          guardrails_modernization, loop_hardening,
│                                          runtime_hardening, agent_eval, agent_identity
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

This is the **recommended learning path**. Each level assumes concepts from the previous one. **73 notebooks across 13 levels**, ordered by increasing complexity. Everything from Level 10 on runs **offline with injected fakes** (🔑 no API keys).

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

### Level 10 · Supervisor & multi-agent (4 notebooks · 🔑 no API keys)

The SUPERVISOR core itself — routing, hierarchy, fan-out — all rebuilt offline with deterministic fakes.

42. [`supervisor_quickstart.ipynb`](notebooks/supervisor_quickstart.ipynb) — supervisor state machine + deterministic intent routing.
43. [`hierarchical_supervisor.ipynb`](notebooks/hierarchical_supervisor.ipynb) — domain supervisors + network routing.
44. [`parallel_research_fanout.ipynb`](notebooks/parallel_research_fanout.ipynb) — map-reduce fan-out/fan-in with `Send`.
45. [`subagent_spawner.ipynb`](notebooks/subagent_spawner.ipynb) — concurrency semaphore, per-task timeouts, `cancel_all()`.

### Level 11 · Specialized agents & memory (6 notebooks · 🔑 no API keys)

Agent subsystems beyond the graph: self-improvement, skills, memory, media ingestion, review pipelines, federated retrieval.

46. [`memory_management.ipynb`](notebooks/memory_management.ipynb) — short-term buffer, Markdown transcripts, long-term store with redaction.
47. [`meta_learner.ipynb`](notebooks/meta_learner.ipynb) — score traces → flag → propose, with a human-review sentinel.
48. [`skill_creator_agent.ipynb`](notebooks/skill_creator_agent.ipynb) — generate a skill offline: GENERATE → VALIDATE → WRITE → REPORT.
49. [`multimodal_ingestion.ipynb`](notebooks/multimodal_ingestion.ipynb) — validate → sanitize → spill → audit media into `AgentState`.
50. [`blind_review_pipeline.ipynb`](notebooks/blind_review_pipeline.ipynb) — spec → implement → two blind reviewers → synthesis → HITL.
51. [`rag/10_federated_rag.ipynb`](notebooks/rag/10_federated_rag.ipynb) — federated multi-node RAG, fault-tolerant merge + re-rank.

### Level 12 · Opt-in phases — Kokoro · Skynet · Budget · A2A (9 notebooks · 🔑 no API keys)

The flag-gated layers: persona deliberation, swarm map-reduce, cost governance, agent-to-agent interop.

52. [`kokoro_deliberation.ipynb`](notebooks/kokoro_deliberation.ipynb) — three souls argue, one judge decides (Fase K).
53. [`skynet_swarm.ipynb`](notebooks/skynet_swarm.ipynb) — plan → `Send` fan-out → workers → reduce → evaluate (Fase S).
54. [`skynet_direct_api.ipynb`](notebooks/skynet_direct_api.ipynb) — the same layer driven directly, plus deterministic re-plan.
55. [`skynet_specialist_swarm.ipynb`](notebooks/skynet_specialist_swarm.ipynb) — specialist roles, metering, a faked remote A2A worker (Fase S+).
56. [`budget_governance.ipynb`](notebooks/budget_governance.ipynb) — `CostMeter` + `BudgetGuard`: within → soft → hard (Phase C).
57. [`budget_graph_integration.ipynb`](notebooks/budget_graph_integration.ipynb) — seed, guard, degrade, abort, clear at the graph seam.
58. [`a2a_server.ipynb`](notebooks/a2a_server.ipynb) — expose prismal as an A2A agent: Agent Card + JSON-RPC/SSE (Phase I).
59. [`a2a_remote_node.ipynb`](notebooks/a2a_remote_node.ipynb) — delegate to a remote agent as a graph node or as tools.
60. [`a2a_tool_provider.ipynb`](notebooks/a2a_tool_provider.ipynb) — remote skills as `a2a__*` tools composed into `CompositeToolProvider`.

### Level 13 · Infrastructure & hardening — Fases X/Y/Z/W/R + hardening phases (13 notebooks · 🔑 no API keys)

The hexagonal ports and the production-hardening layers — how a host wires and defends the whole runtime.

61. [`tool_provider_custom.ipynb`](notebooks/tool_provider_custom.ipynb) — the smallest `ToolProviderPort` (Fase Y).
62. [`tool_provider_host.ipynb`](notebooks/tool_provider_host.ipynb) — host-style composite provider injection (Fase Y).
63. [`vector_store_lancedb.ipynb`](notebooks/vector_store_lancedb.ipynb) — swap the vector-store backend by configuration (Fase Z).
64. [`config_source_env.ipynb`](notebooks/config_source_env.ipynb) — inject the default env config source (Fase W).
65. [`config_source_custom.ipynb`](notebooks/config_source_custom.ipynb) — Vault-style source + per-tenant settings (Fase W).
66. [`composition_root.ipynb`](notebooks/composition_root.ipynb) — `build_runtime` / `build_test_runtime`, tenant isolation (Phase R).
67. [`node_typesafety.ipynb`](notebooks/node_typesafety.ipynb) — node I/O contracts: off / warn / enforce (Phase NTS).
68. [`observability_integration.ipynb`](notebooks/observability_integration.ipynb) — `ObservabilityPort`: run summaries, judge scores, dataset export (Phase OBS).
69. [`guardrails_modernization.ipynb`](notebooks/guardrails_modernization.ipynb) — content-safety reasoning + `StructuredOutputGuard` (Phase GRD).
70. [`loop_hardening.ipynb`](notebooks/loop_hardening.ipynb) — context compaction + phase-narrowed tool catalogue (Phase LH).
71. [`runtime_hardening.ipynb`](notebooks/runtime_hardening.ipynb) — injection, output, tool-policy and runaway guards (Phase H).
72. [`agent_eval.ipynb`](notebooks/agent_eval.ipynb) — `EvalRunner` harness over a scripted graph (Phase V).
73. [`agent_identity.ipynb`](notebooks/agent_identity.ipynb) — DID identities, scoped vault secrets, policy engine, OBO tokens (Phase IDN).

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

**Notebooks without API keys (🔑):** all of Level 1 + 3 + part of Level 2, plus **everything in Levels 10–13** (offline with injected fakes); orchestrators run in simulation mode without network access.

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
