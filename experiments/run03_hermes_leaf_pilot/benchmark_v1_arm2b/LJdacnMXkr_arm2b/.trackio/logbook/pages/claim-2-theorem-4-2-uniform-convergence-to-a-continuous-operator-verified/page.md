## Claim 2 — Theorem 4.2, uniform convergence to a continuous operator — **verified**

Script `verify_claim2.py`, results `results/claim2.json`.

Fixed bandwidth, quadrature-weighted kernel on `[0,1]` and `[0,1]^2`; the "continuous"
operator is the same Sinkhorn fixed point on a 2048-node (1D) / 3136-node (2D)
quadrature, evaluated *at the coarse nodes* (no interpolation error). Sup-norm error of
`T_N f − T_ref f` on a smooth test function:

- Gaussian 1D, eps=0.10: N=32→256; sup errors=2.9e-3 → 7.9e-4 → 2.1e-4 → 5.2e-5; observed rate=1.89, 1.95, 1.99
- exponential 1D, eps=0.10: N=32→256; sup errors=2.2e-3 → 5.7e-4 → 1.4e-4 → 3.5e-5; observed rate=1.96, 1.99, 2.00
- Gaussian 2D, eps=0.15: N=10²→32²; sup errors=1.8e-2 → 7.6e-3 → 3.3e-3 → 1.6e-3; observed rate=0.90, 1.05, 1.27
- exponential 2D, eps=0.15: N=10²→32²; sup errors=1.8e-2 → 6.3e-3 → 2.4e-3 → 1.1e-3; observed rate=1.13, 1.19, 1.32


Uniform (sup-norm) convergence is monotone with clean algebraic rates for both kernel
families, matching the theorem's statement.

**Mutations.** (M1) tying the bandwidth to the spacing (`eps = 2h`) is *not*
discriminating — it still converges to its own near-identity limit; reported honestly.
(M2) a sign-oscillating kernel `cos(|x−y|/eps)` (violating the positivity hypothesis)
makes the Sinkhorn scaling diverge (`NaN` at every resolution), confirming positivity is
load-bearing.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Theorem 4.2, uniform convergence to a continuous operator \u2014 **verified**"}\n-->
