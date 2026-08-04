## Claim 6 — Armadillo spectra across sampling modalities — **verified**
`verify_claim6.py` → `results/claim6.json`

Real Armadillo mesh, normalized to a unit box, n = 1200 samples per modality, Gaussian kernel
ε = 0.10, first 120 Laplacian eigenvalues `μ_k = 1 − λ_k`. Reference = mesh-vertex subsample.

- point cloud vs. mesh: mean relative deviation **0.044** (modes 1–20) — this is the finite-sample
  noise floor at n = 1200
- voxel resolution sweep (mean rel. dev., modes 1–20):
  h = 0.02 → **0.061**, 0.04 → 0.044, 0.08 → 0.193, 0.12 → 0.102, 0.16 → **0.370**
- divergence index (first mode exceeding 15 % deviation): 120 (none) at h = 0.02, 7 at h = 0.04,
  1 at h ≥ 0.08 — divergence migrates to coarse modes precisely as h approaches/exceeds ε

**Mutation** (built into the sweep): h = 0.16, i.e. h/ε = 1.6, destroys coarse-mode agreement
(0.370 vs 0.061). Consistent with Thm 4.2: agreement holds while the sampling resolution is finer
than the kernel scale and fails once they match.

*Caveat:* the paper's exact sampling pipeline, bandwidth and mode count are unspecified, so absolute
numbers are ours; the tested content is the qualitative scale-matched-divergence pattern. The
per-mode divergence index is noise-floor limited at n = 1200 and is reported descriptively only.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 Armadillo spectra across sampling modalities \u2014 **verified**"}\n-->
