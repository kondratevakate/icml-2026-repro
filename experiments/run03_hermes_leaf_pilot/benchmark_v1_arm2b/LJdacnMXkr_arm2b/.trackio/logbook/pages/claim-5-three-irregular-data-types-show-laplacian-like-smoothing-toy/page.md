## Claim 5 — three irregular data types show Laplacian-like smoothing — **toy**

Script `verify_claim5.py`, results `results/claim5.json`.
The jaw-bone voxel scan is not distributed; it is replaced by a **synthetic sparse
hollow curved shell** on a 24³ grid (773 occupied voxels, 75% random occupancy).

Operational test per modality: Dirichlet-energy decay under repeated application of `T`;
frequency-ordered damping (|eigenvalue| monotone in the generator index); generator PSD
and annihilating constants; plus a best-scalar-fit residual against the analytic
Laplacian where that is meaningful.

- point cloud, uniform disk: n=900; Dirichlet energy (k=0→6)=842.9 → 0.108; damping monotone=yes; annihilates constants=2.2e-11; Laplacian fit residual=**0.166**
- sparse voxel shell (jaw proxy): n=773; Dirichlet energy (k=0→6)=730.6 → 0.074; damping monotone=yes; annihilates constants=8.3e-11; Laplacian fit residual=0.999 (n/a: curved surface ⇒ Laplace–Beltrami, not ambient Δ)
- GMM, covariance-aware Mahalanobis kernel: n=300; Dirichlet energy (k=0→6)=247.2 → 0.034; damping monotone=yes; annihilates constants=8.4e-13; Laplacian fit residual=0.786 (n/a: strongly non-uniform density ⇒ density-weighted generator)


All three modalities show the qualitative Laplacian-like behaviour, and the flat
uniform-density case recovers the analytic Laplacian to 17% relative residual.

**Mutation.** A geometry-free random row-stochastic operator gives Laplacian-fit residual
0.60 (vs 0.166 for the real construction) — the test responds to geometry.

**Verdict rationale (`toy`):** behaviour reproduces on all three data *types*, but with
synthetic proxies rather than the paper's assets, so Section-5 figures cannot be matched
numerically.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 three irregular data types show Laplacian-like smoothing \u2014 **toy**"}\n-->
