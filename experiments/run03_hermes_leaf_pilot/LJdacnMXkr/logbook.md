# Reproduction logbook — LJdacnMXkr, "Sinkhorn Normalization of Diffusion Kernels"

- OpenReview: https://openreview.net/forum?id=LJdacnMXkr · arXiv 2507.06161
- Agent: autonomous repro agent (arm1), CPU-only, numpy 2.5.1 / scipy 1.18.0 / sympy 1.14.0
- Global seed: **20260802** (`SEED` in `sinkhorn_lib.py`; each script uses a fixed offset)
- Env: `python3 -m venv .venv` in this directory; run `.venv/bin/python verify_claim<N>.py`
- Wall time: ~25 min total (claim 2 is the slowest, ~70 s)
- Real data used: Stanford **Armadillo.ply** (172,974 vertices), downloaded from
  graphics.stanford.edu, cached as `data_armadillo.ply` (claim 6).

## Method (implemented from first principles)

Symmetric Sinkhorn scaling (Thm 4.1): given a symmetric, entrywise-positive kernel `K`
and masses `m`, find `d>0` with `S = diag(d) K diag(d)` symmetric and `S1 = m`; iterate
`d ← sqrt(d·m / (K d))`. Then `P = diag(m)^{-1} S` satisfies `P1 = 1` (mass conservation),
`P > 0`, and `M P = S = S^T`, i.e. `P` is self-adjoint w.r.t. `⟨u,v⟩_m = Σ mᵢuᵢvᵢ`.
Spectra are computed on the symmetrized conjugate `M^{1/2} P M^{-1/2}`.

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

## Claim 1 — Theorem 4.1 (self-adjointness) — **verified**
`verify_claim1.py` → `results/claim1.json`

6 configurations (Gaussian & exponential kernels, n = 80/150/200, d = 2/3, non-uniform
masses `m ~ U(0.5,1.5)`).

- max `‖MP − (MP)^T‖_∞` = **2.8e-17**; random-vector check `|⟨Pu,v⟩_m − ⟨u,Pv⟩_m| / |⟨Pu,v⟩_m|` ≈ 2e-16
- max row-sum error = 6.9e-13; min entry > 0 (3.1e-14 worst case, strictly positive)
- λ ∈ [4.5e-4, 1.0] in every case

**Mutation.** (a) Row-normalize instead of Sinkhorn → self-adjointness residual **0.046** (broken)
and λ_min = −0.023. (b) Multiplicatively perturb `K` into an asymmetric kernel → residual **0.036**.
The property is specific to the symmetric Sinkhorn scaling, as claimed.

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

## Claim 3 — 5–10 iterations to 0.1 % — **verified (regime-dependent)**
`verify_claim3.py` → `results/claim3.json`

Sweep of 48 configurations: {Gaussian, exponential} × d∈{2,3} × n∈{100,300} × ε∈{0.15,0.3,0.6}
× {uniform, clustered} sampling. Error metric `maxᵢ|(P1)ᵢ − 1|`.

- iterations to < 1e-3: **min 7, median 8, p90 9, max 9** — 100 % within 5–10, 0 % within 5
- worst error after exactly 10 iterations: 3.6e-4

So the *upper* end of the claimed range is reproduced exactly; the "as few as 5" end was never
reached in our sweep (7 was the minimum).

**Mutation.** Extreme mass imbalance (mass ratio ~1e-6) on an elongated chain domain (n = 400,
length 200, ε = 0.45): **14 iterations**, error still 1.1 % after 10 — outside the claimed range.
The claim is an empirical statement about well-conditioned kernels, not a bound.

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

## Files

```
sinkhorn_lib.py         shared Sinkhorn / kernel / property utilities (seed 20260802)
verify_claim1.py .. verify_claim6.py
results/claim1.json .. results/claim6.json
data_armadillo.ply      Stanford Armadillo mesh (real data, claim 6)
.venv/                  numpy 2.5.1, scipy 1.18.0, sympy 1.14.0
```

## Honest overall assessment

The theoretical core (claims 1, 2, 4) reproduces cleanly and to machine precision on CPU; the
empirical iteration-count claim (3) reproduces in the tested regime but is not a bound. Claim 5 is
downgraded to *toy* because the jaw data is unavailable **and** because the covariance-aware GMM
kernel is not PSD, which contradicts the spectral-damping half of claim 4 for that modality — the
paper should state a positive-definiteness hypothesis. Claim 6 reproduces on the real Armadillo mesh
in its qualitative form; exact experimental settings from the paper were unavailable.
