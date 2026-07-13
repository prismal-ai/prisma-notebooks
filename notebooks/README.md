# notebooks/

Executable Jupyter notebooks for the Prismal framework. For the full overview
of the repo (installation, datasets, regeneration), see [`../README.md`](../README.md).

This file is a **quick reference** for the suggested execution order.

## Entry point

Open [`00_index.ipynb`](00_index.ipynb) — lists the 41 notebooks grouped by
category with their dataset and whether they require an API key.

## Recommended order (from simplest to most complex)

**41 notebooks across 9 levels**. Each level assumes concepts from the previous one.

### Level 1 · Framework basics (🔑 no API keys)

1. [`extension/custom_node.ipynb`](extension/custom_node.ipynb)
2. [`extension/custom_subgraph.ipynb`](extension/custom_subgraph.ipynb)
3. [`visualize_graphs.ipynb`](visualize_graphs.ipynb)
4. [`extension/discover_plugins_demo.ipynb`](extension/discover_plugins_demo.ipynb)
5. [`extension/langchain_migration.ipynb`](extension/langchain_migration.ipynb)

### Level 2 · Simple patterns

6. [`patterns/09_parallel_dispatcher.ipynb`](patterns/09_parallel_dispatcher.ipynb) (🔑)
7. [`patterns/06_reflection_loop.ipynb`](patterns/06_reflection_loop.ipynb)
8. [`patterns/02_debate.ipynb`](patterns/02_debate.ipynb)
9. [`patterns/08_swarm.ipynb`](patterns/08_swarm.ipynb)

### Level 3 · Multimodal with mocks (🔑 no API keys)

10. [`multimodal/04_modality_router.ipynb`](multimodal/04_modality_router.ipynb)
11. [`multimodal/05_multimodal_fusion.ipynb`](multimodal/05_multimodal_fusion.ipynb)
12. [`multimodal/01_vision_agent.ipynb`](multimodal/01_vision_agent.ipynb)
13. [`multimodal/02_audio_agent.ipynb`](multimodal/02_audio_agent.ipynb)
14. [`multimodal/03_video_agent.ipynb`](multimodal/03_video_agent.ipynb)
15. [`multimodal_pipeline.ipynb`](multimodal_pipeline.ipynb)

### Level 4 · RAG fundamentals

16. [`rag/01_crag.ipynb`](rag/01_crag.ipynb) — baseline
17. [`rag/05_hybrid_search.ipynb`](rag/05_hybrid_search.ipynb)
18. [`rag/04_hyde.ipynb`](rag/04_hyde.ipynb)

### Level 5 · Advanced patterns

19. [`patterns/07_constitutional_ai.ipynb`](patterns/07_constitutional_ai.ipynb)
20. [`patterns/01_tree_of_thoughts.ipynb`](patterns/01_tree_of_thoughts.ipynb)
21. [`patterns/05_mixture_of_agents.ipynb`](patterns/05_mixture_of_agents.ipynb)
22. [`patterns/04_llm_compiler.ipynb`](patterns/04_llm_compiler.ipynb)
23. [`patterns/03_lats.ipynb`](patterns/03_lats.ipynb)

### Level 6 · Advanced RAG

24. [`rag/06_hierarchical_rag.ipynb`](rag/06_hierarchical_rag.ipynb)
25. [`rag/07_rag_fusion.ipynb`](rag/07_rag_fusion.ipynb)
26. [`rag/03_self_rag.ipynb`](rag/03_self_rag.ipynb)
27. [`rag/08_multi_vector_rag.ipynb`](rag/08_multi_vector_rag.ipynb)
28. [`rag/02_adaptive_rag.ipynb`](rag/02_adaptive_rag.ipynb)
29. [`rag/09_multimodal_rag.ipynb`](rag/09_multimodal_rag.ipynb)

### Level 7 · Simple subgraphs

30. [`subgraphs/04_code_review.ipynb`](subgraphs/04_code_review.ipynb)
31. [`subgraphs/07_document_generation.ipynb`](subgraphs/07_document_generation.ipynb)
32. [`subgraphs/05_data_etl.ipynb`](subgraphs/05_data_etl.ipynb)
33. [`subgraphs/08_debate_consensus.ipynb`](subgraphs/08_debate_consensus.ipynb)
34. [`subgraphs/06_customer_service.ipynb`](subgraphs/06_customer_service.ipynb)

### Level 8 · Advanced subgraphs

35. [`subgraphs/01_ml_pipeline.ipynb`](subgraphs/01_ml_pipeline.ipynb)
36. [`subgraphs/02_financial_analyst.ipynb`](subgraphs/02_financial_analyst.ipynb)
37. [`subgraphs/03_dev_pipeline.ipynb`](subgraphs/03_dev_pipeline.ipynb)
38. [`subgraphs/09_hitl_approval.ipynb`](subgraphs/09_hitl_approval.ipynb)

### Level 9 · Hierarchical orchestrators (most complex)

39. [`subgraphs/12_research_orchestrator.ipynb`](subgraphs/12_research_orchestrator.ipynb)
40. [`subgraphs/11_engineering_orchestrator.ipynb`](subgraphs/11_engineering_orchestrator.ipynb)
41. [`subgraphs/10_analysis_orchestrator.ipynb`](subgraphs/10_analysis_orchestrator.ipynb)

## Per-notebook conventions

1. **Cell 0** — markdown with name, dataset, expected result and original docstring.
2. **Cell 1** — installs/applies `nest_asyncio` for `await main()` in cells.
3. **Cells 2…N** — alternating markdown/code following the sections of the `.py`.
4. **Final cell** — `# await main()` ready to uncomment.

The 4 notebooks that read `data/*.csv` have an extra `_find_data_root()` cell
that locates the `data/` folder automatically.
