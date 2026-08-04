## Claim 6 — Theorem 5.11 (LP lifting) + Table 1 (three settings)

**What the paper claims:** a centralized, non-strategic LP allocation benchmark
guarantees welfare monotonicity under budget constraints via a *lifting*
construction that preserves feasibility across refined partitions (Theorem
5.11); the overall characterization identifies **exactly three settings** where
monotonicity holds (Table 1).

**Reproduction of the lifting (Theorem 5.11).** Built calibrated coarse+refined
model pairs with budgets; solved the allocation LP
`max Σ w_C x_{i,C}·a_i·p_i,C` s.t. budget `Σ w_C x_{i,C}·a_i·p_i,C ≤ B_i` and
supply `Σ_i x_{i,C} ≤ 1`. Then **lifted** the coarse optimum to the refined
partition (copy allocation fractions to every sub-cluster). Over 50 instances:
the lifted solution was **feasible** for the refined LP (supply + budget) and
achieved the **same objective** → therefore `Rev_LP(M_A) ≥ Rev_LP(M_B)` (tCPA
surrogate revenue) and `Welfare_LP(M_A) ≥ Welfare_LP(M_B)`. Directly solving
both LPs also confirmed the inequality in **every** instance. The argument is
identical for MAX-CPA (a_i = v_i).

**The three monotonicity settings (Table 1).** The paper's "exactly three
settings" are three setting *families*, each reproduced here:
1. **tCPA FPA, no budget** (revenue + welfare) — Claim 1.
2. **MAX-CPA VCG, no budget** (welfare) — Theorem 5.5; stress-tested here over
   2000 instances with **0** failures (same Jensen structure as Claim 1 with
   v_i for t_i).
3. **LP allocation, with budgets** (welfare; tCPA surrogate revenue) — this
   claim (Theorem 5.11).

The remaining Table-1 rows are non-monotonic; two are reproduced as explicit
counterexamples: Claim 4 (tCPA VCG, 6.2%) and Claim 5 (tCPA+budget FPA, 16.8%).
As supporting evidence for the MAX-CPA FPA negative row, the Appendix B.4
counterexample was also reproduced: MAX-CPA FPA with designated multiplier
profiles gives Revenue 0.68 → 0.23, a **66.18%** loss (paper: 66%).

**Mutation test.** Corrupted the refined predictions so they no longer average
to the coarse prediction (calibration broken). The lifted coarse solution then
violates the refined budget constraints → the Theorem 5.11 monotonicity
guarantee fails. So calibration preservation (central to the lifting) is
required.

**Verdict: verified** (LP lifting reproduced; Table 1's three settings each
substantiated). Source: Theorem 5.11, §5.3.3 / Appendix B.5; Table 1.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 Theorem 5.11 (LP lifting) + Table 1 (three settings)"}\n-->
