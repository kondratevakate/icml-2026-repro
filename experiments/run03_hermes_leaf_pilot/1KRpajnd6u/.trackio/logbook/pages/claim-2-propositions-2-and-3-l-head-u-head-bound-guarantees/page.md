# Claim 2 — Propositions 2 and 3 (L-head / U-head bound guarantees)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e81f348b968d", "created_at": "2026-08-02T04:00:00+00:00", "title": "Claim 2 \u2014 Propositions 2 and 3 (L-head / U-head bound guarantees)"}
-->
**Paper source:** Section 3.3, Propositions 2 and 3 (with Table 1 parameterizations).
**Script:** `verify_claim2.py` → `results/claim2.json`. Command: `.venv/bin/python verify_claim2.py`.

Heads implemented verbatim from Sec 3.3: `a_i = u_i − ℓ`, `α = σ(·) ∈ (0,1)`,
`π = softmax` over the K stencil directions, `F_{i→i+d} = a_i α_i π_{i→i+d}`; and dually
`b_i = u_max − u_i`, `β = σ(·)`, `ρ = softmax` over incoming directions,
`F_{j→i} = b_i β_i ρ_{j→i}` with the sender-side outflow recovered by the inverse roll.

- **Symbolic:** worst-case (zero inflow) L-head margin `u^{t+1} − ℓ = a(1−α) > 0`;
  worst-case (zero outflow) U-head margin `u_max − u^{t+1} = b(1−β) > 0`.
- **Numeric:** 4 shapes (1D and 2D, incl. non-power-of-two 127 and 48×24) × 3 radii ×
  15 seeds × 3 logit scales (σ = 1, 5, **20** — the last saturates sigmoid/softmax to the
  adversarial extreme) = **540 configs**, 20 steps each, and every third seed starts with 10 %
  of cells sitting exactly *on* the bound.
  - L-head: **0/540** violating configs, worst excursion 3.55e-15.
  - U-head: **0/540** violating configs, worst excursion 1.11e-15.
  Both worst values are float64 roundoff (< 1e-12), not structural violations.
- **No post-hoc correction:** the implementation is asserted to contain no `clip`/`maximum(`
  call (`no_clipping_in_implementation: true`), so the bound comes from the parameterization.

**Mutation test:** replace the capacity fraction `σ(·) ∈ (0,1)` by `1.6·σ(·)`, which can exceed
the available amount / remaining capacity — i.e. invert the one inequality the proofs rely on.
Result: **120/120** configs violate, max violation 340.0 (L) and 342.0 (U). The guarantee is
therefore attributable to the `α, β < 1` capacity fraction, exactly as the proofs state.
**Verdict: verified.**
