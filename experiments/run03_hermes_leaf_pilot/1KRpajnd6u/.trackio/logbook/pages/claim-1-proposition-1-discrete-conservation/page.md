# Claim 1 — Proposition 1 (discrete conservation)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d825d58976a7", "created_at": "2026-08-02T04:00:00+00:00", "title": "Claim 1 \u2014 Proposition 1 (discrete conservation)"}
-->
**Paper source:** Section 3.2, Eq. (2); Appendix A, Proposition 1, Eqs. (6)–(9).
**Script:** `verify_claim1.py` → `results/claim1.json`. Command: `.venv/bin/python verify_claim1.py`.

The proposition holds for *any* flux field, so instead of training a network I sampled head
outputs (logits) randomly — a strictly stronger test than one trained model.

- **Symbolic (exhaustive, exact rationals):** 1D periodic ring, radius-1 symmetric stencil,
  fully symbolic fluxes. `sum(u^{t+1}) - sum(u^t)` simplifies to `0` for N = 3, 4, 5, 6.
- **Numeric:** 4 shapes × 3 radii × 20 seeds = **240 configs** per dtype, 20 rollout steps each,
  L-head fluxes, periodic `np.roll` inflow.
  - float64: max relative drift **4.436e-16**, median 2.18e-16 (exact to machine epsilon).
  - float32: max **7.392e-08**, median 1.475e-08, min 1.85e-09.
    The paper reports 1.19e-7 / 2.38e-7 / 1.79e-7 (Table 2, float32) and 3.3e-8 / 7.7e-8
    (Tables 3–4). **The observed float32 band coincides with the reported one**, confirming that
    the reported "machine precision" numbers are exactly float32 accumulation roundoff, not a
    property of the trained model.

**Mutation tests (mechanism, not correlation):**
1. *asymmetric_inflow* — receiver credited 1.001× the amount the sender debited (breaks the
   shared-flux property while keeping everything else): drift **9.68e-3 … 1.04e-2**, i.e.
   ≥1.3e5× the unmutated float32 error.
2. *nonperiodic* — zero-padded shift instead of periodic roll (breaks the symmetric-stencil /
   periodic-BC hypothesis): drift **2.26e-2 … 4.52e-1**, ≥3e5× baseline.

Both mutations fail in exactly the direction Proposition 1 predicts. **Verdict: verified.**
