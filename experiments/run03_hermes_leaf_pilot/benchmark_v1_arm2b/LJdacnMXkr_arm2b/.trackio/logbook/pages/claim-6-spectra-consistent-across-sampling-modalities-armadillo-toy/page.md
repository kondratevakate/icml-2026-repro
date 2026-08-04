## Claim 6 — spectra consistent across sampling modalities (Armadillo) — **toy**

Script `verify_claim6.py`, results `results/claim6.json`.
The Armadillo mesh is unavailable; the proxy shape is the **unit sphere**, whose exact
Laplace–Beltrami spectrum `l(l+1)` (multiplicity `2l+1`) provides absolute ground truth.
Three modalities, n=900 each, eps=0.25: uniform random surface points; Fibonacci "mesh
vertices"; voxelised shell (22³ snap, re-projected).

Accuracy against the exact sphere spectrum (modes 1–9, normalised by `λ(l=1)`):

- uniform point cloud: mean rel. err=12.2%; max rel. err=28.1%
- mesh vertices (Fibonacci): mean rel. err=5.1%; max rel. err=14.7%
- voxelised shell: mean rel. err=8.8%; max rel. err=22.3%


Individual eigenvalues inside a degenerate multiplet are split arbitrarily by sampling
noise, so the distribution-level comparison uses multiplet group means (blocks l=1,2,3):

- uniform point cloud (ref): l=1=1.000; l=2=2.880; l=3=5.283; max deviation vs point cloud=—
- mesh vertices: l=1=1.000; l=2=2.812; l=3=5.121; max deviation vs point cloud=3.1%
- voxelised shell: l=1=1.000; l=2=2.825; l=3=5.190; max deviation vs point cloud=1.9%
- exact sphere: l=1=1.000; l=2=3.000; l=3=6.000; max deviation vs point cloud=—


Deviation grows with mode index (l=1: 0%, l=3: 1.8–3.1%) — i.e. divergence appears at the
finer scales, consistent with the claim and with Theorem 4.2.

**Mutations.** A mild anisotropic stretch (z × 1.35) does **not** break the normalised
group-mean statistic (max dev 0.5%) — reported as a non-discriminating mutation.
Changing the surface to a torus (R=1, r=0.45) does: max group deviation **11.0%**,
above the 10% consistency threshold.

**Verdict rationale (`toy`):** the qualitative phenomenon reproduces cleanly on a proxy
shape with known ground truth, but not on the paper's Armadillo asset.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 spectra consistent across sampling modalities (Armadillo) \u2014 **toy**"}\n-->
