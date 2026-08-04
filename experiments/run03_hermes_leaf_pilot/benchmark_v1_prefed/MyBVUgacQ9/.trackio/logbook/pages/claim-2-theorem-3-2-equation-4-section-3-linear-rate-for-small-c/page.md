# Claim 2 — Theorem 3.2 + Equation 4 (Section 3): linear rate for small ‖C‖

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_310a8a0354c6", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 2 \u2014 Theorem 3.2 + Equation 4 (Section 3): linear rate for small \u2016C\u2016"}
-->
**Verdict: `verified` (qualitative content of eq. 4; the constants m1,m2,m3 are not numerically
specified in the paper, so the exact threshold cannot be tested).**

*Source:* Theorem 3.2 and Equation (4), Section 3. *Script:* `verify_claim2.py` → `results/claim2.json`.

Setup: toy problem with the nonlinear part scaled, `s·(x1x2 − x3x4) + q z + 1 = 0`, q = 2 fixed
(so Q is fixed, as eq. 4 requires), ‖C‖ = s swept; 48 initialisations × 3 ρ per cell; the
limit point's reduced Hessian was checked (second-order differentiability / local-minimum part).

| ‖C‖ | best-ρ worst-case contraction factor | regime |
|---|---|---|
| 0 (linear constraints) | **0.0588** | Thm 3.2 region (reference) |
| 0.01 / 0.05 / 0.2 / 0.5 | 0.0589 / 0.0589 / 0.0588 / 0.0588 | Thm 3.2 region |
| 1 | **0.1111** | Thm 3.2 region |
| 2 | 0.2507 | mutation arm |
| 5 | 0.8058 | mutation arm |
| 10 | 0.8585 | mutation arm |
| 25 | **0.9981** | mutation arm |
| 50 | **1.0004** (no contraction; final violation 7.7e-05) | mutation arm |

So over the whole small-‖C‖ range the rate is linear and essentially identical to the ‖C‖ = 0
(purely linear-constraint) case — exactly the claim "as long as ‖C‖ is small enough, linear
convergence is still preserved", including the ‖C‖ → 0 consistency with Lin et al. (2015b).

**Mutation (mechanism: eq. 4).** Inflating ‖C‖ past the small regime degrades the factor
monotonically and destroys linear convergence entirely at ‖C‖ = 25–50 (factor ≥ 0.998, and at
‖C‖ = 50 the iterates no longer even reach feasibility, violation 7.7e-05). The rate therefore
tracks the ‖C‖-vs-Q balance that eq. (4) posits, rather than being a property of ADMM per se.

---
