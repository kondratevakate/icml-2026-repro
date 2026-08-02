# Claim 5 — improvements on heterogeneous-noise (Cor 5.2), linear (Cor 5.4), unimodal (Cor 5.7)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3a2dd5dfde8a", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 5 \u2014 improvements on heterogeneous-noise (Cor 5.2), linear (Cor 5.4), unimodal (Cor 5.7)"}
-->
**Source:** Section 5 (5.1 Alg 5 / Thm 5.1 / Cor 5.2; 5.3 Cor 5.4; 5.4 Cor 5.7).
**Script:** `verify_claim5.py` → `results/claim5.json`
**Verdict: `verified` for the heterogeneous-noise application; `inconclusive` for the linear and
unimodal applications.** (Reported as `partial` in the JSON.)

### 5a Heterogeneous noise — `verified`
Instance: Gaussian, `mu = [1.0, 0.6, 0.4, 0.2]`, `sigma = [0.15, 0.15, 1.0, 1.0]`, giving
`A = **295.78**`, `C = **956.13**` from Theorem 5.1.
- **Theorem 5.1** (Algorithm 5 implemented verbatim, 5 seeds × 400 runs per δ ∈ {0.2, 0.1, 0.05}):
  `0/3` violations of `P(τ > T*_δ) ≤ δ` and `0/3` of δ-correctness.
- **FC2FB(PE-KHN) end-to-end** (real algorithm inside the real meta-algorithm, δ₀=1/e, Q=1,
  5 seeds × 200 runs per budget): mean error `0.724 → 0.050 → 0.0070 → 0.0 → 0.0` for
  `B = 2500, 5000, 10 000, 20 000, 40 000`, monotone decreasing and **0 violations** of the
  Theorem 3.2 bound (`2.488, 2.127, 1.588, 0.919, 0.329`).
- **K⁷ vs K⁹ separation** (sympy, exact): on the paper's instance the FC2FB(PE-KHN) complexity is
  `K**7 − 2K**6 + 2K**5` (degree **7**) and SHVar's is `K**9 − 2K**8 + 2K**5` (degree **9**),
  matching the paper's `O(K⁷)` vs `O(K⁹)`.
- **Mutation test:** make the allocation noise-blind (use σ_max for every arm — the SH/SHVar
  style rule). The sample budget grows by **4.85×** (δ=0.1) and **4.78×** (δ=0.01) on the same
  instance. The per-arm σ_i allocation is the source of the improvement.

**Additional finding — Corollary 5.2 is printed stronger than its parent theorem.** Theorem 3.2
with `δ₀ = 1/e, Q = 1` yields denominator `4 + 4A·log₂B`; Corollary 5.2 prints `4 + 4A·lnB`.
Since `log₂B = 1.4427·lnB`, the printed corollary bound is strictly *smaller* than what
Theorem 3.2 gives, at all **9/9** checked `(A, B)` points; the gap reaches a factor
**3.29e28** at `A=10, B=1e5`. This looks like a `ln`↔`log₂` typo in the corollary; it does not
affect Theorem 3.2, and the empirical errors above satisfy both forms.

### 5b Linear bandits (Corollary 5.4) — `inconclusive`
Stated in terms of `gamma*`, `rho*` and the algorithm *Fixed Budget Peace* (Katz-Samuels et al.
2020, Algorithm 3). None of these is defined in this paper; verifying the corollary requires
re-implementing that external algorithm and computing its instance constants. Not attempted
within budget. **No toy substitute was constructed** — a self-invented "linear-bandit-ish"
simulation would test my own construction, not the paper's claim.

### 5c Unimodal bandits (Corollary 5.7) — `inconclusive`
Same reason: stated via `T_mu(delta)` and *UniTT* (Poiani et al. 2024, Theorem 3.7), external.
Partial evidence does exist: the **FCW2S half** of the Corollary 5.7 pipeline is verified in
Claim 3 with exactly the parameters Corollary 5.7 prescribes (`δ₁ = 1/(8e)`,
`L = ⌈4 ln(1/δ)/ln(1/(4e δ₁))⌉`), and the FC2FB half is verified in Claim 1. What is unverified is
the instance-dependent constant `T_mu(δ₁)+K` and the claimed dominance over `Δ^{-2}` (Prop 5.8).

---
