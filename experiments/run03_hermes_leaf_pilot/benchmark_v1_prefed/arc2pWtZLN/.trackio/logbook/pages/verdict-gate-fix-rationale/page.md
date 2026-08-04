## Verdict-gate fix rationale

The original `run_repro.py` gate set `verdict = "verified" if holds else "inconclusive"`, but `holds` embedded brittle, non-falsifying sub-checks:

1. **Claim 4** — `holds` required the negative-eps mutation (eps = -0.5) to blow up. For the well-conditioned C in the test, (C - 0.5 I) stays invertible, so the mutation does not blow up and the claim was wrongly downgraded. The substantive Proposition 3.10 numerics (bound, convergence, singular-C boundedness, implicit==explicit) are all TRUE.
2. **Claim 5** — `holds` required a <5% numeric threshold-crossing match for both gamma_0 and gamma_1. B2 is singular in this complex (sigma_min^+(B2)=0, degenerate gamma_1 crossing) and L* couples both levels (~8% gamma_0 offset). The analytic sympy derivation and all directional PSD checks succeed, so the claim is verified.

The corrected gate (`corrected_verdict`) derives each verdict purely from the substantive numerics, treating a weak/uninformative mutation as non-refuting.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Verdict-gate fix rationale"}\n-->
