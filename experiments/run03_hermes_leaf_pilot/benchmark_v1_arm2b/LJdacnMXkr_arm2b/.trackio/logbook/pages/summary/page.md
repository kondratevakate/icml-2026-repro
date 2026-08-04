## Summary

| # | Claim (source) | Verdict |
|---|---|---|
| 1 | Symmetric smoothing operator can be diagonally rescaled (symmetric Sinkhorn) into a diffusion operator self-adjoint w.r.t. a mass-weighted inner product (Theorem 4.1) | **verified** |
| 2 | Gaussian/exponential Sinkhorn-normalized operators converge uniformly on bounded domains to continuous diffusion operators as resolution increases (Theorem 4.2) | **verified** |
| 3 | Symmetric Sinkhorn needs only 5–10 iterations to push normalization error below 0.1% (Section 4/5) | **verified** |
| 4 | Normalized operators jointly satisfy symmetry, mass conservation, positivity and spectral damping — properties not jointly satisfied by row- or symmetric-normalization (Theorem 4.1) | **inconclusive** |
| 5 | Method demonstrated on point clouds, sparse voxel grids (jaw bone) and covariance-aware GMM kernels, showing Laplacian-like smoothing (Section 5) | **toy** |
| 6 | Armadillo shape analysis: spectra consistent across sampling modalities, divergence only at scales matching sampling resolution (Section 5) | **toy** |

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary"}\n-->
