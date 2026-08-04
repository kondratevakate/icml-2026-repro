# Claim 5 — Proposition 1: PT is a special case of localized CP; IS flags such methods

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_166ff82ef43a", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 5 \u2014 Proposition 1: PT is a special case of localized CP; IS flags such methods"}
-->
**Source:** Proposition 1 (§3.2), Definition 1 (§4), Remarks 3 and 12. **Script:** `verify_claim5.py` → `results/claim5.json`.
**Verdict: `verified`.**

Justifying numbers (10 seeds × p ∈ {0.93, 0.95, 0.96, 0.98} × σ̂-floor ε ∈ {1e-2 … 1e-12}, α=0.10):
- With σ̂(x) ∈ {ε, 1} drawn as {1−p, p}, localized CP's normalized-score quantile Q converges to the base
  score quantile at α′ = 1−(1−α)/p: worst-case gap over all seeds/p at ε ≤ 1e-9 is **0.201** on a half-width
  of ≈11.8 (1.7 %), and per-test-point interval lengths differ from the PT reference by at most **0.403** out of
  ≈22 — the entire residual is the finite-sample quantile-index effect (Q is taken as the ⌈(n+1)(1−α)⌉-th smallest
  of n scores of which (1−p)n are +∞, versus the ⌈(m+1)(1−α′)⌉-th of the m = p·n finite ones); it vanishes as n→∞.
  Structurally the intervals are identical: μ̂(x) ± Q·ε ≈ null point-set w.p. 1−p, μ̂(x) ± Q w.p. p — Eq (2).
- This "cherry-picked scale estimator" localized CP is **shorter than VCP on every one of the 40 seed×p cells**
  (`all_localized_shorter_than_vcp: true`), while carrying no task information.
- Interval Stability flags it: IS ≥ **10.08** in every cell (e.g. p=0.93 → 37.30, p=0.98 → 10.48), whereas
  length/coverage alone would rate the method superior.

**Mutation test:** replace the random two-point σ̂ with the deterministic σ̂ ≡ 1. Localized CP then collapses
to VCP exactly — mean length equals the VCP length to < 1e-9 on all 40 cells — and IS falls to **2.8e-26 ≈ 0**.
The flag therefore fires on the randomness, not on the localization.

---
