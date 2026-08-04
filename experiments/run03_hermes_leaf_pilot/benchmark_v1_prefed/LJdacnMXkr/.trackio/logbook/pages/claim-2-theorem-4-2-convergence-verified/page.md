## Claim 2 — Theorem 4.2 (convergence) — **verified**
`verify_claim2.py` → `results/claim2.json`

Domain [0,1], quadrature masses `mᵢ = 1/N`, fixed bandwidth ε = 0.08, test function
`f = sin 3x + x²`. Continuum reference = same construction at N = 6400. Sup error of
`(P_N f − f)/ε²` over 41 interior points:

| N | 100 | 200 | 400 | 800 | 1600 | rate |
|---|---|---|---|---|---|---|
| Gaussian | 2.09e-2 | 5.21e-3 | 1.30e-3 | 3.20e-4 | 7.63e-5 | **2.02** |
| Exponential | 1.11e-2 | 2.81e-3 | 7.02e-4 | 1.74e-4 | 4.13e-5 | **2.02** |

Uniform (sup-norm) convergence at ~O(N⁻²) for both kernel families.

**Mutation.** Tie the bandwidth to the sampling scale (ε = 4/N): the sup error to the fixed
continuum operator plateaus (Gaussian 0.698 → 0.624 → 0.624 → 0.624; exponential grows to 10.8),
i.e. no continuum limit — the theorem's bandwidth/resolution separation is necessary.

*Caveat:* 1D, uniform sampling, one smooth test function. The general-domain / random-sampling
version of Thm 4.2 is only partially covered.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Theorem 4.2 (convergence) \u2014 **verified**"}\n-->
