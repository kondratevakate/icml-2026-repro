## Claim 4 — joint properties vs. standard normalizations — **verified**
`verify_claim4.py` → `results/claim4.json`

6 configurations, 3 schemes, four boolean properties (self-adjointness < 1e-8, mass error < 1e-6,
positivity, spectrum ⊂ [0,1]):

| scheme | symmetry | mass | positivity | spectrum | all four |
|---|---|---|---|---|---|
| Sinkhorn `diag(d)K diag(d)` | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| row `D⁻¹K` | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| symmetric `D^{-1/2}KD^{-1/2}` | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 |

Exactly the failure pattern the paper asserts: row normalization loses self-adjointness (and picks
up λ_min ≈ −0.023, λ_max ≈ 1.035), symmetric normalization loses mass conservation.

**Mutation.** Symmetric, entrywise-positive but **indefinite** kernel (`0.5(A+Aᵀ)+0.05`, A uniform):
Sinkhorn still gives symmetry / mass / positivity but **λ_min = −0.066**. Spectral damping therefore
relies on positive-definiteness of the kernel (true for Gaussian/exponential), which the claim
statement does not make explicit.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 joint properties vs. standard normalizations \u2014 **verified**"}\n-->
