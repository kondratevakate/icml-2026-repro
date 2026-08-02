# Claim 1 — Theorem 4.2 (FC-RCGP-UCB), sublinear regret for α < 1/4

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c19447c00229", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 1 \u2014 Theorem 4.2 (FC-RCGP-UCB), sublinear regret for \u03b1 < 1/4"}
-->
**Source:** Theorem 4.2, Sec 4.3; sublinearity discussion Sec 4.3 "Conditions for Sublinear Regret",
Appendix D.4.3, Table 1 (Appendix E).
**Verdict: verified** (as an implication: stated bound ⇒ stated α threshold).

Substituting T_c = T^α, Ψ(T_c) = √(1+T_c(1+T_c)) (Sec 4.1, κ = σ_noise² = 1), γ_T = Õ(1),
β'_T = Õ(1) into R_T = Õ(Ψ(T_c)(1+√T_c)√(β'_T T(γ_T+T_c))) gives, symbolically (sympy),

* growth exponent **e(α) = 2α + 1/2**, hence e(α) < 1 ⟺ **α < 1/4** (`alpha_threshold_rbf = 1/4`);
* this coincides exactly with the paper's intermediate form R_T = Õ(T_c²√T)
  (`intermediate_matches_bound = true`);
* numeric exponents: α=1/10 → 0.6999, α=1/5 → 0.9000, α=1/4 → 1.0000, α=1/3 → 1.1667 (not sublinear),
  α=1/2 → 1.5000;
* all **8 Matérn/domain-case entries of Table 1 for FC** reproduce to grid resolution 1/4000
  (e.g. η=1/3, cases 1&2: 0.222 vs 2/9; η=1/2, case 3: 0 vs 0 — no permissible α).

**Mutation test:** removing the observability penalty Ψ (set Ψ≡1) changes the exponent to α + 1/2
and the threshold to **α < 1/2**, which is exactly the paper's "Ideal" column in Table 1
(`matches_paper_ideal = true`, `verdict_changed = true`). The threshold therefore genuinely comes
from Ψ, not from an artefact of the algebra.
