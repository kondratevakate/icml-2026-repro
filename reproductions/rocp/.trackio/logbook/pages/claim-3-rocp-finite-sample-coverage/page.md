# Claim 3: ROCP-finite-sample-coverage


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_079f711467fa", "created_at": "2026-07-28T15:58:37+00:00", "title": "Claim 3: ROCP-finite-sample-coverage"}
-->
**Anchored claim (verbatim).** "Risk-Optimal Conformal Prediction (ROCP) is introduced as a practical algorithm that targets these risk-minimizing sets while maintaining finite-sample, distribution-free marginal coverage."

**Verdict -- VERIFIED.**

`audit_claim3_exchangeability.py` implements Algorithm 1 literally: for every candidate test label, calibration includes that candidate in the `(n+1)` coverage constraint. It enumerates every selector breakpoint rather than assuming coverage is monotone in beta.

Across a nine-observation support, all exchangeability orbits for `n+1` in `{3,4,5,6}`, and three alpha values per size, **14,850 orbit/alpha checks pass**. Worst coverage reaches the finite-sample bound exactly up to `1.11e-16`. Because every finite exchangeable distribution is a mixture over permutation orbits, this is an exhaustive finite-support check, not a random toy sample.

**Official-code comparison.** Against official commit `3ee0cf6e393d2e434368bc1fc7fe3abd03ed493f`, the audit reproduces two paper/code divergences:

1. Released `calibrate_beta()` omits the candidate test point and uses one global beta, while Algorithm 1 calibrates candidate-wise on `n+1` observations.
2. At `t=0`, Remark 3.2 gives `C(x,0)=Y`; released code can return a strict subset.

Concrete finite cases produce different official and paper-faithful prediction
sets. These are implementation divergences, not a falsification of the paper.
The full cached medical and driving checks below complete the empirical gate.

**Fresh empirical coverage.** The released evaluator was rerun on all 20
COVID seeds under both loss matrices and all 20 BDD splits. Across the seven
alpha values, the largest absolute difference between mean ROCP miscoverage
and alpha was `0.00244`, below half a percentage point. This empirical check
uses the released global-beta approximation; the distribution-free guarantee
itself is assigned from the independent literal Algorithm 1 audit above.
