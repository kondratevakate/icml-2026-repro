## Claim 5 — three irregular data modalities — **toy**
`verify_claim5.py` → `results/claim5.json`

n = 600 per modality. Point cloud = bumpy sphere; voxel = synthetic mandible-arch sparse voxel grid
(surrogate); GMM = 6-component mixture with covariance-aware kernel
`k = exp(−½ dᵀ(Σᵢ+Σⱼ)⁻¹d/ε²)`.

| modality | mass err | Dirichlet energy monotone | smooth gain | noise gain | λ_min |
|---|---|---|---|---|---|
| point cloud | 6e-14 | yes | 0.848 | 0.200 | 0.000 |
| sparse voxel (jaw surrogate) | 6e-14 | yes | 0.985 | 0.230 | 0.000 |
| GMM covariance-aware | 6e-14 | yes | 0.309 | 0.075 | **−0.146** |

Laplacian-like low-pass behaviour (`L1 = 0` to 1e-13, monotone Dirichlet-energy decay under `P^k`,
smooth component preserved and noise damped) reproduces on all three data types.

**Mutation.** Scramble the coordinates used to build the kernel: smooth-signal gain drops
0.848 → 0.089 and corr(Pf, smooth) 0.967 → 0.308. The low-pass behaviour is genuinely geometric.

**Why "toy", not "verified":** (a) the paper's jaw-bone voxel data is not public, so that experiment
is only a synthetic surrogate; (b) the covariance-aware GMM kernel is **not** positive definite, so
its Sinkhorn operator violates the spectral-damping property claimed in Claim 4 (λ_min = −0.146).
That is a genuine tension between claims 4 and 5 as stated.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 three irregular data modalities \u2014 **toy**"}\n-->
