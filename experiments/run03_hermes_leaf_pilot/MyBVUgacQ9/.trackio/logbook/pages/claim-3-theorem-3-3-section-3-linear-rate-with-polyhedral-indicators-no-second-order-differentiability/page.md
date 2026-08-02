# Claim 3 — Theorem 3.3 (Section 3): linear rate with polyhedral indicators, no second-order differentiability

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_068d6021ae27", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 3 \u2014 Theorem 3.3 (Section 3): linear rate with polyhedral indicators, no second-order differentiability"}
-->
**Verdict: `verified`.**

*Source:* Theorem 3.3, Section 3. *Script:* `verify_claim3.py` → `results/claim3.json`.

Setup: toy problem + **box** constraints `x_i ∈ [0.3, 1.5]` (boxes are polyhedra) chosen so that
the limit sits on an active face; 12 seeds per cell; the Theorem-3.3 quantity
`L(x^k,z^k,w^k) − min_{(x,z)∈B((x^k,z^k);r)} L(x,z,w^k)` with r = 0.05 was evaluated exactly by
SLSQP over box ∩ ball.

| cell (q, ρ) | worst residual factor | worst Thm-3.3 gap factor | active constraints at limit | local-min probe failures |
|---|---|---|---|---|
| q=2, ρ=1 | **0.2000** | **0.0522** | 4 / 4 | 0 / 12 |
| q=2, ρ=4 | 0.0668 | **0.0881** | 4 / 4 | 0 / 12 |
| q=5, ρ=1 | 0.0385 | 0.0056 | 4 / 4 | 0 / 12 |
| q=5, ρ=4 | 0.0273 | 0.0818 | 4 / 4 | 0 / 12 |

All four box constraints are active at every limit point, i.e. the Lagrangian is **not**
second-order differentiable there and Theorem 3.2 does not apply — yet the Theorem-3.3 gap decays
geometrically with per-iteration factor ≤ 0.0881, and 4×12×400 = 19 200 feasible perturbation
probes found **no** better point than the limit (local minimality, second part of Thm 3.3).

**Mutation A (eq. 4 broken while keeping polyhedrality).** ‖C‖×10 → residual factor 0.9858 and
gap factor 0.9977; ‖C‖×50 → residual factor 1.0006 (no contraction at all). The polyhedral
structure alone does not buy the rate; eq. (4) is still needed, as the theorem states.

**Mutation B (polyhedrality broken, eq. 4 kept).** Replacing the box by a Euclidean ball
(exact projection) still yields a geometric factor ≈ 0.0625 across 12 seeds. This is reported
as-is: Theorem 3.3 is a sufficient condition, and our evidence does not show polyhedrality to be
necessary — it only shows the theorem's conclusion holds inside its stated hypothesis.

---
