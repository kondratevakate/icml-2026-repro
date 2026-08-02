# Claim 2 — inverted sample complexity `O(A ln(1/δ)·ln(A ln(1/δ)/Q) + C)`

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d2640ec65f0b", "created_at": "2026-08-02T05:00:00+00:00", "title": "Claim 2 \u2014 inverted sample complexity `O(A ln(1/\u03b4)\u00b7ln(A ln(1/\u03b4)/Q) + C)`"}
-->
**Source:** Section 3, the display immediately after Theorem 3.2 (δ₀ = 1/e).
**Script:** `verify_claim2.py` → `results/claim2.json`
**Verdict: `verified`.**

Method: exact integer bisection for `B_req(δ)` = smallest budget satisfying both the Theorem 3.2
validity condition and `bound ≤ δ`; compared against
`F = Q ln(1/δ) + A ln(1/δ) ln(max(e, A ln(1/δ)/Q)) + C`. Deterministic, no randomness.

Justifying numbers, over **420** grid points (`A ∈ {1,10,10²,…,10⁶}`, `C ∈ {0,10,10³,10⁵}`,
`Q ∈ {1,10,100}`, `δ ∈ {0.1,10⁻²,10⁻³,10⁻⁶,10⁻¹⁰}`):
- `max B_req/F = **24.41**` — a single absolute constant across 6 orders of magnitude in A,
  5 in C and 10 in 1/δ. The per-A maximum is **non-increasing**: `24.41, 24.40, 24.14, 20.94,
  15.45, 13.66, 12.63` for `A = 1 … 10⁶` (growth factor A=10⁶ over A=10 is **0.518 < 1**).
  That is exactly what the `O(·)` asserts.

**Mutation test.** Drop the inner logarithm (`F_naive = Q ln(1/δ) + A ln(1/δ) + C`, i.e. claim
that FB matches FC *without* log factors). The ratio then **grows monotonically with A**:
`24.41, 56.93, 83.38, 105.62, 126.89, 147.83, 168.58` for `A = 1 … 10⁶` — a factor **6.91**
blow-up, unbounded in A. So the `ln(A ln(1/δ)/Q)` factor is necessary and is not padding.

Diagnostic (not part of the verdict): in the recommended regime `Q ≤ A`, `max B_req/T*_δ = 168.6`
and `max B_req/(T*_δ (1+ln T*_δ)²) = 13.6`, i.e. the overhead is polylog, not polynomial.

---
