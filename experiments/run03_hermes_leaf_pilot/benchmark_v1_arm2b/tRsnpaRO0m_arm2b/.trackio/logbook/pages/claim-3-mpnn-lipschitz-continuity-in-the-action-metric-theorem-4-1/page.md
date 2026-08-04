## Claim 3 — MPNN Lipschitz continuity in the action metric (Theorem 4.1)

Script `verify_claim3.py` → `results/claim3.json`. Verdict: **verified**. Mutation test: yes.

Setup: `n=200`, fiber bound `r=4`, depth `D=3`, layer `h ← tanh(a·h + b·A h)` with `a=b=0.5`.
tanh is 1-Lipschitz, so the theorem's constant is bounded by
`C_theory = L^D + b·Σ_{k<D} L^k = 20.5` with `L = a + b·r = 2.5`.
The operator term of `d_M`, `sup_{‖g‖≤1}‖(A₁−A₂)g‖`, is computed **exactly** as the spectral norm of
`A₁−A₂` (one SVD) — never sampled. Two probes over a 6-point signal-scale sweep (0.02 … 5.0):

| probe | max empirical ratio | ≤ `C_theory = 20.5`? |
|---|---|---|
| P1 signal-only perturbation (`A₁=A₂`) | **4.195** | yes |
| P2 joint graph (double-edge swaps) + signal perturbation | **0.157** | yes |

**Criterion note (legitimate revision, documented).** A first iteration checked *scale-invariance*
of the ratio; that encoded a **wrong definition** of Lipschitz. With tanh the ratio *decreases* as
signal scale grows (saturation), so the correct criterion is `max_ratio ≤ C_theory`, not a constant
ratio. The criterion was wrong, not the claim.

**Mutation.** Dense Erdős–Rényi `p=0.3` (fiber mass ≈ 0.3n ≫ r=4) gives max ratio **94.63 > 20.5** —
the bofop constant is violated, showing the *bounded fiber mass is exactly what makes the Lipschitz
constant finite*.

**Caveat.** The paper gives no numeric value for `C'_{D,r}`, so only the **form** of the inequality
(and the necessity of the fiber bound) is checkable; it holds with room (4.20 vs 20.5).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 MPNN Lipschitz continuity in the action metric (Theorem 4.1)"}\n-->
