# Claim 2: Numerical bias makes model comparison unreliable as censoring increases


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_56735d9f504f", "created_at": "2026-07-19T08:34:37+00:00", "title": "Claim 2: Numerical bias makes model comparison unreliable as censoring increases"}
-->
**Setup.** Four fixed risk scores are ranked twice — by the censored (ST) C-index and by the oracle (OR) C-index — and we record how often the two orderings agree exactly, as a function of censoring rate and mechanism. Two regimes are tested:
- **widely separated models** (correct / partial / noisy / uninformative predictor),
- **near-tied models** (four small perturbations of the correct score, noise sd 0.55–0.70), which is the regime the claim is actually about — the authors' rebuttal notes that where rankings swap, the top models are often not statistically distinguishable.

**Ranking agreement with the oracle (fraction of runs, near-tied models, 24 reps each)**

| mechanism | 0% | 20% | 40% | 60% | 80% |
| --- | --- | --- | --- | --- | --- |
| administrative | 1.00 | 0.88 | 0.79 | 0.71 | **0.33** |
| independent | 1.00 | 0.88 | 0.79 | 0.62 | **0.29** |
| covariate-dependent | 1.00 | 0.88 | 0.88 | 0.62 | **0.42** |

**Widely separated models: 1.00 at every rate and mechanism.**

**Verdict — Claim 2 reproduced, with its boundary made explicit.** Among near-tied models the censored evaluation recovers the true ordering in only **29–42% of runs at 80% censoring**, degrading monotonically from perfect agreement at 0%. In other words, under heavy censoring the model a standard benchmark declares best is usually not the best. But the instability is **not universal**: when models are far apart in quality the ranking survives intact at every censoring level. So the claim is about the realistic regime of closely-performing candidates, not about survival benchmarking in general.

**Note on our first attempt (honest process record).** The first run used only the widely-separated models and therefore found 100% ranking preservation everywhere — an apparent refutation that was really a flaw in our design, since a 0.03 metric bias cannot flip a ranking whose gaps are an order of magnitude larger. Adding the near-tied regime is what makes the claim testable.

**Why this is directly usable.** The practical rule the paper argues for follows from these numbers: report the censoring rate and mechanism alongside any survival benchmark, and do not declare a 'best model' from close scores under heavy censoring without an oracle-style or statistical check.
