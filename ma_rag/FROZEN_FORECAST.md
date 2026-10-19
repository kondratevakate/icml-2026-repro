# MA-RAG frozen forecast

Frozen: 2026-07-29. OpenReview `S7lpdz7NAW`; arXiv `2603.03292v3`.

Written after primary paper/repository inventory and before independent dataset
counts, mutation tests, or runtime checks. Leaderboards and third-party
reproduction verdicts are excluded.

| Claim | Expected |
|---|---:|
| C1. MA-RAG implements a multi-round solver → conflict-guided retrieval → history-ranking loop with consensus early stopping. | 2/2 possible from independent code-routing checks and dependency-light unit tests. |
| C2. The official release covers the seven declared medical QA benchmarks with valid multiple-choice schemas and ground-truth answers. | 2/2 expected from independent JSON integrity, ID, option, answer, and overlap audits. |
| C3. MA-RAG-ext averages 62.2% versus 55.4% for Qwen3-8B, a +6.8 point gain across seven benchmarks. | 0/2 expected: no canonical predictions, full MedCorp corpus/index, evaluator checkpoint, API endpoint, or local Qwen3-8B serving stack is frozen. |
| C4. MA-RAG-ext exceeds the strongest RAG and test-time scaling baselines by the reported margins. | 0/2 expected for the same missing matched-run artifacts. |
| C5. Multi-Refine, Retrieval Agent, and Ranking Agent contribute cumulative gains to 58.7, 60.6, and 62.2 average accuracy. | 0/2 expected without canonical ablation outputs and evaluator data/checkpoint. |
| C6. MA-RAG has lower four-run variance and favorable inference efficiency on challenging benchmarks. | 0/2 expected without four matched stochastic runs, hardware/service configuration, and timing traces. |

Frozen expectation: **4/12**; plausible range **2–4/12**. Stop empirical
claims if MedCorp, model/API services, evaluator checkpoint, or original
predictions are missing. Paper tables and the repository result image count
only as source concordance.
