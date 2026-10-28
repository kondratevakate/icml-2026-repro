# scRatio frozen forecast

Frozen: 2026-07-29. OpenReview `5zbPdMNcl9`; arXiv `2602.24201v2`.

Written after primary paper/source and official-artifact inventory, before
cached-result recomputation, notebook-output auditing, dataset inspection, or
claim scoring. Leaderboards and third-party verdicts are excluded. A
dependency smoke test and an analytic-oracle exercise had already established
that the released ratio-ODE path imports and can recover a known Gaussian
ratio; this makes C1 less blind than the other forecasts.

| Claim | Expected |
| --- | ---: |
| C1. Proposition 4.1 and its single-ODE ratio estimator are implemented correctly. | 2/2 expected from source tracing and an analytic-oracle check. |
| C2. scRatio improves Gaussian ratio MSE and inference runtime. | 1-2/2 expected: scripts and cached result tables exist, but a paper-scale retraining is GPU-heavy. |
| C3. scRatio is best or second-best on the five-dimensional MI sweep. | 1/2 expected from executed notebooks/tables; fresh 100k-sample multi-run training is unlikely locally. |
| C4. scRatio leads the semi-synthetic PBMC differential-abundance metrics. | 1/2 expected if processed data and executed outputs are internally complete; full model/baseline refits are expensive. |
| C5. scRatio detects lower batch signal after scVI correction on two datasets. | 1/2 expected from released data/notebooks and independent aggregation, without full scVI/scRatio refits. |
| C6. scRatio captures ComboSciPlex and donor-specific cytokine effects. | 1/2 expected from released outputs and targeted checks; both studies are qualitative and large-scale. |

Frozen expectation: **7/12**; plausible range **6-8/12**. Full credit beyond C1
requires fresh repeated training or sufficiently complete frozen predictions
with independently reconstructible metrics.
