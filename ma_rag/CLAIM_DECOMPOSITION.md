# MA-RAG claim decomposition

- **C1 architecture:** independently trace candidate generation, consensus
  stopping, conflict-query generation, retrieval, and history ranking in the
  released entropy and evaluator variants.
- **C2 released benchmarks:** validate all seven JSON files for record counts,
  IDs, questions, options, answers, duplicates, and cross-file overlap.
- **C3 headline gain:** requires fresh matched predictions for Qwen3-8B and
  MA-RAG-ext across all seven benchmarks.
- **C4 baseline superiority:** requires identical backbones, corpus, retriever,
  reranker, prompts, sampling budget, and evaluation splits.
- **C5 ablation:** requires canonical cumulative Multi-Refine, Retrieval Agent,
  and Ranking Agent runs.
- **C6 variance/efficiency:** requires four independent stochastic runs and
  matched hardware timing traces.
