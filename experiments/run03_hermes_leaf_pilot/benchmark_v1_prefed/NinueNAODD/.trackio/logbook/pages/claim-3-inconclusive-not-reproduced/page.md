## Claim 3 — — inconclusive (not reproduced)

**Verdict:** table 1

*Source:* Table 1, IHQ-unscr. top-1. *Script:* `verify_claim3.py` (19 min, 500 rater-bootstraps).

| | BBQ | Crowd-BT | Bayes-BT |
|---|---|---|---|
| task spec | 61.92 | 33.15 | 24.32 |
| arXiv v2 Table 1 | 61.42 | 32.44 | 23.59 |
| **ours** | **44.20** | **71.40** | **23.20** |

- Bayes-BT lands almost exactly on the paper's value (23.2 vs 23.59). BBQ is well below, and **our Crowd-BT is the most stable top-1 identifier — the opposite of the claim's ordering.**
- Confounders that stop this being a refutation: smaller public data sample; substitute gt; our Crowd-BT is a reimplementation whose stability is hyperparameter-dependent (lr 0.02, 20 epochs); 500 instead of 10,000 bootstraps.
- **Mutation:** re-drawing every pair's winner by a fair coin collapses all three to 4.0%, i.e. chance level (1/28 = 3.57%). ✔ the metric is measuring real signal.
- Version note: the task spec's numbers (61.92/33.15/24.32) are **not** the v2 numbers (61.42/32.44/23.59) — they come from an earlier version of the paper.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 \u2014 inconclusive (not reproduced)"}\n-->
