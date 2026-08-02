# Claim 2 — Theorem 4.3 (A2-RCGP-UCB), sublinear regret for α < 1/7

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0628518f0178", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 2 \u2014 Theorem 4.3 (A2-RCGP-UCB), sublinear regret for \u03b1 < 1/7"}
-->
**Source:** Theorem 4.3, Sec 4.3; Appendix D.5.5; Table 1 (Appendix E).
**Verdict: verified** (same implication sense as claim 1).

R_T = Õ((1+T_c Ψ(T_c)²)√(β'_T T(γ_T+T_c))) with T_c = T^α gives exponent **e(α) = 7α/2 + 1/2**,
so sublinear ⟺ **α < 1/7 = 0.142857…**; matches the paper's intermediate R_T = Õ(T_c^{7/2}√T).
Numeric: α=1/7 → 1.0000, α=1/6 → 1.0833 (not sublinear), α=1/4 → 1.3750, α=1/3 → 1.6667.
All 8 Matérn entries of Table 1 for A2 reproduce (grid 1/7000), incl. η=1/4 case 3: 0.08329 vs 1/12.
**Mutation test:** Ψ≡1 ⇒ exponent 3α/2 + 1/2 ⇒ **α < 1/3**, again exactly the paper's "Ideal" column.
