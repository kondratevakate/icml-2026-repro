## Claim 3 — Theorem 5.2 / Corollary 5.3 (μ=1 optimality + welfare)

**What the paper claims:** uniform bidding at μ=1 is simultaneously
conversion-maximizing, revenue-maximizing among equilibria, and
welfare-maximizing for tCPA bidders in FPA. Welfare monotonicity follows as a
corollary (Assumption 3.1: v_i = t_i ⇒ welfare = revenue for tCPA).

**Reproduction.** (1) Feasibility: CPA per won cluster = μ·t_i, so μ≤1
required; at μ=1 the CPA constraint binds. (2) Revenue `Rev(μ)=μ·Σ_C w_C
max_i t_i·p_i,C` is monotone increasing in μ and maximized at the feasible
boundary μ=1 (mean revenue over 1000 instances: μ=0.3→0.648, 0.5→1.080,
0.7→1.512, 0.9→1.944, 1.0→2.159). (3) Welfare at μ=1 equals
`Σ_C w_C max_i t_i·p_i,C`, i.e. the allocation `argmax_i t_i·p_i,C` which is
welfare-maximizing (value = t_i). (4) Welfare-monotonicity corollary: over 2000
refined model pairs, `Welfare(M_A) ≥ Welfare(M_B)` in **every** instance
(follows because welfare == revenue for tCPA).

**Mutation test.** (a) μ=0.5 yields strictly lower revenue than μ=1
(property shifts). (b) Setting v_i ≠ t_i breaks the Welfare == Revenue identity
(observed in 1000/1000 instances), so the Corollary-5.3 premise fails.

**Verdict: verified.** Source: Theorem 5.2 and Corollary 5.3, §5.2.1.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Theorem 5.2 / Corollary 5.3 (\u03bc=1 optimality + welfare)"}\n-->
