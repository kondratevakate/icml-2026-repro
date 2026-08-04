# Reproduction logbook — LJdacnMXkr, "Sinkhorn Normalization of Diffusion Kernels"

- Arm: **arm2** (Hermes + K-Dense skill set, no context compaction)
- Agent model: `tencent/hy3:free` via `localhost:8319/v1`
- Hardware: CPU only (numpy 2.5.1 / scipy 1.18.0 / sympy 1.14.0), global seed `20260803`
- Paper assets: **no code, no data release** with the submission; `paper/` was not present in
  the run bundle. All reproductions are **first-principles re-implementations** of the
  described construction (symmetric Sinkhorn scaling of a symmetric positive kernel),
  with the paper's real assets (jaw-bone voxel scan, Armadillo mesh) replaced by proxies
  where noted.
- Shared implementation: `sinkhorn_lib.py`; per-claim drivers `verify_claim<N>.py`;
  raw numbers in `results/claim<N>.json`.

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

## Claim 1 — Theorem 4.1, diagonal rescaling → mass-weighted self-adjointness — **verified**

Script `verify_claim1.py`, results `results/claim1.json`.

Construction: given symmetric positive `W` and a prescribed positive mass vector `m`,
iterate `d ← sqrt(d·m /(W d))` until `diag(d)W diag(d)1 = m`; set `S = diag(d)W diag(d)`,
`T = diag(m)^{-1} S`.

- Gaussian 2D, n=120: Sinkhorn iters=42; row-sum err=8.7e-13; `<Tx,y>_m − <x,Ty>_m` (rel)=2.2e-16; `T = diag(a)W diag(b)` residual=8.0e-17
- exponential 3D, n=100: Sinkhorn iters=42; row-sum err=5.2e-13; `<Tx,y>_m − <x,Ty>_m` (rel)=0.0; `T = diag(a)W diag(b)` residual=9.8e-17
- Gaussian 5D, n=80: Sinkhorn iters=42; row-sum err=5.1e-13; `<Tx,y>_m − <x,Ty>_m` (rel)=0.0; `T = diag(a)W diag(b)` residual=1.3e-16


(Iteration counts here are to machine tolerance 1e-12, not the 0.1% target of Claim 3.)

**Mutations.** (M1) breaking kernel symmetry: m-self-adjointness error rises to
1.3e-2 / 1.3e-3 / 2.8e-2. (M2) replacing the Sinkhorn diagonal by row normalization:
error 3.2e-1 / 6.1e-2 / 1.7e-1. Both break the property → the test is discriminating.

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

## Claim 3 — 5–10 Sinkhorn iterations for <0.1% error — **verified**

Script `verify_claim3.py`, results `results/claim3.json`.
40 configurations (dim 2/3 × Gaussian/exponential × eps ∈ {0.3,0.5,0.8,1.2,2.0} ×
uniform/non-uniform mass, n=200 each), error metric
`max_i |(diag(d)W diag(d)1)_i − m_i| / m_i`, threshold 1e-3, start `d = 1`.

- iterations: **min 6, median 9.5, mean 9.0, max 11**
- 90% of configurations land in the claimed 5–10 window; 90% are ≤ 10 (the two misses
  need 11).

**Mutations.** Clustered geometry with tiny bandwidth: 9 iterations (not discriminating —
symmetric Sinkhorn is robust there). Extreme target-mass dynamic range (1e-3…1e3):
**14 iterations**, outside the claimed window — the "5–10" figure is bandwidth/mass
dependent, not universal.

## Claim 4 — four properties jointly, and baseline exclusivity — **inconclusive**

Script `verify_claim4.py`, results `results/claim4.json`.

For a *prescribed* non-uniform mass measure `m` (n=150, Gaussian 2D/3D, exponential 2D):

- Sinkhorn operator: **4/4** in every case (sym. rel. err ≤ ~1e-16, mass err ≤ 1e-12,
  entrywise positive, spectrum in [0,1]).
- Row normalization `D^{-1}W` scored against the prescribed `m`: **3/4** (not
  m-self-adjoint, rel. asymmetry 0.24).
- Symmetric normalization `D^{-1/2}WD^{-1/2}`: **2/4** (mass error 0.42, not
  m-self-adjoint).

**Why not "verified":** the primary content holds, but the exclusivity sub-clause does
not survive as literally stated. Row normalization evaluated against **its own degree
measure** satisfies all four properties simultaneously (sym. err 2.2e-16, mass err
4.4e-16, positive, spectrum [5.4e-4, 1.0]) — a genuine counter-example recorded in the
results file. The Sinkhorn advantage is that it attains the four properties for an
*arbitrary prescribed* mass measure, which is a weaker statement than the claim.

**Mutation.** An entrywise-positive but indefinite symmetric kernel
(`1 + 0.9 cos(π|x−y|²)`, min eigenvalue −13.1) yields a Sinkhorn operator with
eigenvalue −0.110 → spectral damping fails. Positive-definiteness of the kernel, not
just positivity of entries, is required for the [0,1] spectrum.

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

## Threats to validity

- No original code or data: every number here comes from an independent re-implementation
  of the described method, so a mismatch could reflect a different construction rather
  than a false claim. Claims 1–4 are construction-intrinsic and largely
  implementation-independent; claims 5–6 are asset-dependent and hence `toy`.
- Claim 3's iteration count depends on bandwidth and on the dynamic range of the target
  mass (14 iterations at 10⁶ mass ratio), so "5–10" holds for benign regimes only.
- Sphere/annulus/torus scale: n ≤ 900 dense eigendecompositions on CPU; finer resolutions
  were not run within budget.
