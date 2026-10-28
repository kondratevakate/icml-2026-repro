# Claim 1: Dual-variable coupling removes steady-state bias (Sec. 3)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f98842a437ca", "created_at": "2026-07-20T08:31:34+00:00", "title": "Claim 1: Dual-variable coupling removes steady-state bias (Sec. 3)"}
-->
**Claim (verbatim, anchored):** "The method (DC-PnPDP) introduces explicit ADMM dual variables that act as an integral controller accumulating historical constraint-violation feedback, in contrast to standard memoryless plug-and-play operators, to reduce steady-state reconstruction bias (Section 3, dual-variable coupling)."

**Setup.** A known r=8-dimensional linear subspace M of R^60 (orthonormal basis U, projector P = UU^T), an exact proximal denoiser `prox_shrink(v) = Pv + s(I-P)v` for a softness parameter s (s to 0 = idealized hard projection onto M, larger s = softer/more realistic denoiser), M_MEAS=40 linear measurements y = Ax_true + noise with x_true in M. Dual-coupled ADMM backbone (Algorithm 1):

    x^{k+1} = argmin_x ||Ax-y||^2 + lambda||x - z^k + u^k||^2
    z^{k+1} = prox_shrink(x^{k+1} + u^k, s)
    u^{k+1} = u^k + (x^{k+1} - z^{k+1})

vs. the loose-coupled baseline (identical iteration with u frozen at 0, i.e. no dual memory). lambda=1.0, 4000 iterations, 12 seeds. Metric: `dist(x,M)/||x||` at convergence. Code: `claim1_manifold.py` (numpy, CPU).

**Results — median distance-to-manifold at convergence, 12 seeds:**

| s | dual-coupled | loose (u=0) |
| --- | --- | --- |
| 0.5 | 9.6e-03 | 1.3e-02 |
| 0.1 | 2.3e-03 | 1.0e-02 |
| 0.01 | 2.5e-04 | 9.7e-03 |
| 0 (idealized) | 3.9e-16 | 9.6e-03 |

Consistent across all 12 seeds, no overlap between the two schemes at any s (verified per-seed, not just on the median). The loose-coupled baseline has a bias floor around 1e-2 regardless of how soft/hard the denoiser is; dual coupling removes that floor and its residual distance scales down as O(s), reaching machine precision (3.9e-16) at s=0.

**Verdict — CONFIRMED cleanly, 12/12 seeds.** This is exactly what the claim asserts: the dual variable u acts as accumulated (integral) feedback of the constraint violation x-z, and this accumulated feedback is what eliminates the steady-state bias that the memoryless (loose-coupled) baseline exhibits regardless of denoiser softness. No caveats on this claim; the effect is large, monotone in s, and has zero overlap between conditions on every one of the 12 seeds.
