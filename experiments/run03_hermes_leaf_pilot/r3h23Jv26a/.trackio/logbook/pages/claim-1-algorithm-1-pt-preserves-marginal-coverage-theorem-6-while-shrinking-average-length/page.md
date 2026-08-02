# Claim 1 — Algorithm 1 (PT) preserves marginal coverage (Theorem 6) while shrinking average length

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_120b28c99aa2", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 1 \u2014 Algorithm 1 (PT) preserves marginal coverage (Theorem 6) while shrinking average length"}
-->
**Source:** §3.2 Algorithm 1 / Eq (2); Theorem 6 (§3.3); length side: Lemma 1, Theorem 10, Corollary 3 (§3.4).
**Type:** theory + small simulation. **Script:** `verify_claim1.py` → `results/claim1.json`.
**Verdict: `verified`.**

Justifying numbers:
- Symbolic (sympy): `p·(1−α′) = 1−α` with α′ = 1−(1−α)/p — identity holds exactly (`symbolic_identity_holds: true`).
  Hence PT's marginal coverage equals the base's nominal level; over the whole grid α∈{.05,.1,.15,.2,.3} ×
  p∈{.90,…,.99} with p>1−α the maximum |PT coverage − (1−α)| is **0.0** (exact enumeration, 26 cells).
- Finite-sample Monte Carlo (misspecified linear fit on Gaussian-mixture-noise data, **20 seeds**, n=2000/fold):
  | α | p | VCP cov | VCP len | PT cov | PT len |
  |---|---|---|---|---|---|
  |0.10|0.96|0.9018|22.704|**0.9006**|**22.294**|
  |0.10|0.98|0.9018|22.704|0.9015|22.489|
  |0.20|0.96|0.8026|21.782|0.8002|21.169|
  |0.20|0.98|0.8026|21.782|0.8009|21.475|
  PT coverage ≥ nominal − 2·SEM in all four cells; PT length < VCP length in all four cells.

**Mutation test 1 (break the mechanism):** drop the α′ adjustment (use α inside the non-null branch).
Coverage collapses exactly as predicted to ≈ p(1−α): 0.9006 → **0.8663** at α=.1, p=.96 (predicted 0.864),
0.8002 → 0.7706 at α=.2, p=.96 (predicted 0.768). Analytic grid deficit up to **0.085**. Coverage is therefore
carried by the α′ adjustment, not by the randomization per se.

**Mutation test 2 (break the sufficient condition):** replace the misspecified Gaussian-mixture DGP with a
well-specified Gaussian one — paper's failure case, Example 3. PT then gets **longer**: 3.312 → 3.594 (α=.1, p=.96),
2.588 → 2.682 (α=.2, p=.96), while coverage stays at 0.902/0.805. The analytic Example-3 inequality
Φ⁻¹(1−α/2) < p·Φ⁻¹((1+(1−α)/p)/2) was checked at 300+ (α,p) points and holds everywhere.
So the length gain is conditional on the concavity/misspecification condition, exactly as the paper states.

---
