# Claim 4 — FC2AT (Algorithm 6): anytime variant via the doubling trick, no budget knowledge

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_74b4737863d6", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 4 \u2014 FC2AT (Algorithm 6): anytime variant via the doubling trick, no budget knowledge"}
-->
**Source:** Section 4 pointer + Appendix D (Algorithm 6, Defs D.1–D.2, Props D.3–D.4, Thm D.5).
**Script:** `verify_claim4.py` → `results/claim4.json`
**Verdict: `verified`.**

Justifying numbers:
- **No horizon knowledge (structural):** phase lengths are `T_i = 2^i·Q`
  (`Q=1 → 2,4,8,16,32,64,128`), a function of `(i, Q)` only; a standing recommendation exists
  at every `t` (checked for all `t ∈ [1, 5000)`).
- **Props D.3 / D.4 exhaustively:** over 4 parameter settings and **80 004 consecutive integer
  horizons T** in the admissible range `T ≥ max(4B*−2Q, 2Q)`:
  **0 violations of `T_{I_f} ≥ B*`** and **0 violations of `T_{I_f} ≥ T/4`**.
- **Theorem D.5:** exact anytime error of the last completed phase vs
  `3exp(−T/(16Q/ln(1/δ₀)+16log₂(T/Q)A))`: **0 violations** over 21 (setting, T) pairs; the error
  falls by at least a factor **4.29e9** across the T range in every setting.

**Mutation test.**
- M1 — remove the doubling (constant phase length `T_i = 2Q`): the error **stalls at 1.0** and
  violates the Theorem D.5 anytime bound at **4 of 7** horizons.
- M2 — sub-geometric growth `T_i = ⌈1.05^i Q⌉`: Prop D.4 (`T_{I_f} ≥ T/4`) fails at **6 of 6**
  horizons. Geometric growth is exactly what buys the anytime guarantee.

---
