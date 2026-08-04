## Verdict summary

| # | Claim (source) | Verdict | Key number |
|---|---|---|---|
| 1 | Diagonal Sinkhorn rescaling ⇒ self-adjoint diffusion operator in mass-weighted inner product (Thm 4.1) | **verified** | self-adjointness residual ≤ 2.8e-17, mass error ≤ 6.9e-13, all entries > 0 |
| 2 | Uniform convergence to a continuous diffusion operator as resolution ↑, Gaussian & exponential (Thm 4.2) | **verified** | sup-error decays at rate ≈ 2.02 in N; rel. error 4.4e-5 (Gauss) / 4.1e-6 (exp) at N=1600 |
| 3 | 5–10 Sinkhorn iterations for < 0.1 % normalization error (Sec. 4/5) | **verified** (regime-dependent) | 48/48 configs converge in 7–9 iters (median 8) |
| 4 | Joint symmetry + mass conservation + positivity + eigenvalues in [0,1]; not jointly held by row/sym normalization (Thm 4.1) | **verified** | Sinkhorn 4/4 in 6/6 configs; row-norm 0/6, sym-norm 0/6 |
| 5 | Works on point clouds, sparse voxel (jaw) grids, GMM covariance-aware kernels (Sec. 5) | **toy** | mechanism holds on all 3; jaw data not public; GMM kernel breaks the [0,1] spectrum |
| 6 | Armadillo spectra consistent across sampling modalities; divergence only at the sampling scale (Sec. 5) | **verified** | coarse-mode dev. 0.061 at h=0.02 → 0.370 at h=0.16 (h/ε=1.6) |

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Verdict summary"}\n-->
