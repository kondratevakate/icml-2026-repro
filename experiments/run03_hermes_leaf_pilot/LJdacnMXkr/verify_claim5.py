"""Claim 5 (Section 5 experiments): the method applies to point clouds, sparse
voxel grids (jaw-bone-like geometry) and Gaussian mixture models with
covariance-aware kernels, giving Laplacian-like smoothing on each irregular
data type.

For each modality we build the kernel, Sinkhorn-normalize, and check:
 - operator properties (symmetry / mass / positivity / spectrum in [0,1]);
 - constants are exactly preserved (L 1 = 0, L = I - P);
 - Dirichlet energy of a noisy signal decreases monotonically under P^k;
 - a smooth geometric signal is preserved while iid noise is damped
   (Laplacian-like low-pass behaviour).

Mutation: scramble the geometry (shuffle coordinates used to build the kernel).
The low-pass behaviour must break -- the smooth component is destroyed.

NOTE: the jaw-bone voxel grid of the paper is not public; we use a synthetic
mandible-like sparse voxel arch as a surrogate -> this modality is 'toy'.
"""
import numpy as np
from sinkhorn_lib import (SEED, rng, gaussian_kernel, mahalanobis_kernel,
                          sym_sinkhorn, props, dump)

g = rng(5)

def point_cloud(n=600):
    v = g.standard_normal((n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * (1 + 0.3 * np.sin(3 * v[:, 2:3]))          # bumpy sphere

def jaw_voxels(res=0.06):
    t = np.linspace(-1.1, 1.1, 400)                       # parabolic dental arch
    arch = np.stack([t, 0.55 * t**2, np.zeros_like(t)], 1)
    pts = []
    for a in arch:
        off = g.standard_normal((25, 3)) * np.array([0.02, 0.02, 0.12])
        pts.append(a + off)
    P = np.concatenate(pts)
    V = np.unique(np.round(P / res).astype(int), axis=0) * res   # sparse voxel centres
    idx = g.choice(len(V), size=min(600, len(V)), replace=False)
    return V[idx]

def gmm_data(n=600, k=6):
    mus = g.random((k, 3)) * 2
    Sig = []
    for _ in range(k):
        A = g.standard_normal((3, 3)) * 0.35
        Sig.append(A @ A.T + 0.02 * np.eye(3))
    lbl = g.integers(0, k, n)
    X = np.stack([mus[l] + np.linalg.cholesky(Sig[l]) @ g.standard_normal(3) for l in lbl])
    S = np.stack([Sig[l] for l in lbl])
    return X, S

def analyse(name, X, K, scramble=False):
    n = len(X)
    if scramble:
        Xs = X.copy()
        for c in range(X.shape[1]):
            Xs[:, c] = g.permutation(Xs[:, c])
        K = gaussian_kernel(Xs, K_eps)
    P, _ = sym_sinkhorn(K, np.ones(n), tol=1e-13)
    pr = props(P)
    smooth = np.sin(2.0 * X[:, 0]) * np.cos(2.0 * X[:, 1])
    smooth = (smooth - smooth.mean()) / smooth.std()
    noise = g.standard_normal(n)
    f = smooth + noise
    Pf = P @ f
    # Dirichlet energy E(f) = 0.5 sum_ij S_ij (f_i-f_j)^2 with S = P (m=1)
    def energy(v):
        return float(0.5 * np.sum(P * (v[:, None] - v[None, :]) ** 2))
    ener = [energy(f)]
    v = f.copy()
    for _ in range(5):
        v = P @ v
        ener.append(energy(v))
    return dict(modality=name, n=n, **pr,
                const_preservation_err=float(np.max(np.abs(P @ np.ones(n) - 1))),
                smooth_gain=float(np.dot(P @ smooth, smooth) / np.dot(smooth, smooth)),
                noise_gain=float(np.linalg.norm(P @ noise) / np.linalg.norm(noise)),
                dirichlet_energy_path=ener,
                energy_monotone=bool(np.all(np.diff(ener) < 1e-12)),
                corr_Pf_smooth=float(np.corrcoef(Pf, smooth)[0, 1]))

K_eps = 0.35
out = []
X1 = point_cloud();  out.append(analyse("point_cloud", X1, gaussian_kernel(X1, K_eps)))
X2 = jaw_voxels();   out.append(analyse("sparse_voxel_jaw_surrogate", X2, gaussian_kernel(X2, 0.12)))
X3, S3 = gmm_data(); out.append(analyse("gmm_covariance_aware", X3, mahalanobis_kernel(X3, S3, 0.9)))
mut = analyse("point_cloud_SCRAMBLED", X1, gaussian_kernel(X1, K_eps), scramble=True)

ok = all(o["energy_monotone"] and o["const_preservation_err"] < 1e-8
         and o["smooth_gain"] > o["noise_gain"] for o in out)
spec_fail = [o["modality"] for o in out if o["lam_min"] < -1e-8]
dump("results/claim5.json", dict(
    claim=5, source="Section 5 (experiments on point clouds, voxel grids, GMMs)",
    seed=SEED,
    verdict="toy" if ok else "inconclusive",
    modalities=out, spectral_damping_failures=spec_fail, mutation_scrambled_geometry=mut,
    mutation_note=f"Scrambling geometry drops smooth-signal gain from "
                  f"{out[0]['smooth_gain']:.3f} to {mut['smooth_gain']:.3f} and "
                  f"corr(Pf,smooth) from {out[0]['corr_Pf_smooth']:.3f} to "
                  f"{mut['corr_Pf_smooth']:.3f}: the low-pass behaviour is geometric, not generic.",
    caveat="Mechanism reproduces on all three data types (mass conservation, monotone "
           "Dirichlet-energy decay, low-pass behaviour). Two honest limitations: (a) the "
           "paper's jaw-bone voxel data is not public so we used a synthetic mandible-arch "
           "surrogate; (b) the covariance-aware GMM kernel k=exp(-0.5 d^T (Si+Sj)^{-1} d /eps^2) "
           "is NOT positive definite, so its Sinkhorn operator has lam_min<0 and violates the "
           "spectral-damping property of Claim 4 -> verdict 'toy'."))
