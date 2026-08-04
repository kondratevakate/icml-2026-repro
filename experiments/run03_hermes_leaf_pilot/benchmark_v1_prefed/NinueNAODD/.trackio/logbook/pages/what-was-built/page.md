## What was built

| file | role |
|---|---|
| `bbq_core.py` | From-first-principles BBQ EM (Eqs. 11–13), Bayes-BT (EM with `q≡1`), Crowd-BT (Chen et al. 2013-style SGD), Eq. (3) log-likelihood, Gamma/Beta log-prior, Kendall's tau, synthetic generator from Eq. (2) |
| `ihq_data.py` | Loader for the real IHQ dataset (`Mabyduck/CLIC2024-test-human-eval`, screened/unscreened parquet, downloaded from HF) |
| `verify_claim1..6.py` | One verifier per anchored claim; each writes `results/claim<N>.json` |
| `results/claim<N>.json` | Numeric results, protocol, mutation test, verdict |
| `data/{screened,unscreened}.parquet` | Real IHQ data (only external data used) |

Hyperparameters throughout are the paper's (Appendix B): Gamma prior `a=5, b=0.1`, Beta prior `α=10, β=2`, `ELO = 400·log λ`, convergence when no Elo score moves by more than 1.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "What was built"}\n-->
