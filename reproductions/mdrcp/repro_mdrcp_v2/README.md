# MDRCP theorem audit bundle

Paper: *Multi-Distribution Robust Conformal Prediction*
(`Vv4XRZDMM0`, arXiv `2601.02998`).

This CPU-only bundle independently audits two claims omitted from the first
`kondratevakate` reproduction:

- Theorem 2, Section 3.1, Eqs. (4)-(5): primal/dual equality, the
  `h_lambda > 1` superlevel structure, boundary randomization, and
  complementary slackness.
- Theorem 3, Section 3.2, Eqs. (6)-(7): convergence of the estimated shared
  score and calibrated max-p set, plus a frozen-weight destructive control
  that violates the theorem's consistency assumption.

Run:

```powershell
python run_all.py
```

The scripts generate deterministic JSON under `results/`. They do not use
another participant's outputs. Public peer logbooks were used only to identify
which claims had a demonstrated full-verdict path; the implementations and
measurements here are independent.

Claim 1 remains supported by the earlier 10-seed classification/regression
coverage experiment. Claims 4 and 5 require a fresh run of the official
paper-scale Linear suite. Claim 6 requires FMoW and is outside this local
bundle.
