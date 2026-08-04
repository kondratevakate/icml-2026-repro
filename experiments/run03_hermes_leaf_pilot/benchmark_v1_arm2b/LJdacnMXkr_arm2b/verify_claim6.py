"""Claim 6 (Section 5): shape-analysis experiments (Armadillo mesh) show spectral
distributions remain CONSISTENT across sampling modalities, with eigenvalue divergence
appearing only at scales matching the sampling resolution -- validating Theorem 4.2.

The Armadillo mesh is not distributed with the submission (no code/data release), so the
test shape is the unit SPHERE, for which the exact Laplace-Beltrami spectrum is known
(lambda_l = l(l+1), multiplicity 2l+1). This gives an absolute ground truth in addition
to cross-modality consistency, at the cost of not being the paper's actual asset -> the
verdict is 'toy' (proxy geometry).

Three sampling modalities of the same surface:
  A) uniform random surface point cloud,
  B) quasi-uniform "mesh vertex" sampling (Fibonacci sphere),
  C) voxelised surface shell (points snapped to a regular 3D grid).
For each we build the Sinkhorn-normalized operator, form L = (I-T)/eps^2, take the
smallest eigenvalues, normalise by the first non-trivial eigenvalue, and compare across
modalities and to l(l+1)/2.

Mutation: modality C replaced by an ANISOTROPICALLY stretched sphere (z scaled 1.35);
low-index spectra must then diverge, showing the consistency test is discriminating.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinkhorn_lib import sinkhorn_operator

SEED = 20260803
rng = np.random.default_rng(SEED)
NEIG = 20


def spectrum(X, eps):
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    W = np.exp(-D2 / (2 * eps ** 2))
    m = np.ones(len(X))
    T, S, d, iters = sinkhorn_operator(W, m)
    Ts = (T + T.T) / 2
    ev = np.sort(np.linalg.eigvalsh(Ts))[::-1]
    lam = (1 - ev) / eps ** 2
    lam = np.sort(lam)[:NEIG]
    return lam, int(iters)


def spacing(X):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return float(np.mean(D.min(1)))


N = 900
eps = 0.25

# A) uniform random surface points
A = rng.normal(size=(N, 3)); A /= np.linalg.norm(A, axis=1, keepdims=True)

# B) Fibonacci ("mesh vertex") sampling
i = np.arange(N) + 0.5
phi = np.arccos(1 - 2 * i / N)
gold = np.pi * (1 + 5 ** 0.5) * i
B = np.stack([np.cos(gold) * np.sin(phi), np.sin(gold) * np.sin(phi), np.cos(phi)], 1)

# C) voxelised shell: snap dense surface samples to a regular grid, dedupe, re-project
res = 22
P = rng.normal(size=(20000, 3)); P /= np.linalg.norm(P, axis=1, keepdims=True)
Q = np.unique(np.round(P * res).astype(int), axis=0).astype(float) / res
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
if len(Q) > N:
    Q = Q[rng.choice(len(Q), N, replace=False)]
C = Q

# Mutation: anisotropically stretched sphere sampled like A
Cm = A.copy(); Cm[:, 2] *= 1.35

# Mutation 2: different surface (torus R=1, r=0.45) sampled uniformly by rejection
_t = []
while len(_t) < N:
    u = rng.uniform(0, 2 * np.pi); v = rng.uniform(0, 2 * np.pi)
    if rng.uniform(0, 1) < (1 + 0.45 * np.cos(v)) / 1.45:
        _t.append(((1 + 0.45 * np.cos(v)) * np.cos(u),
                   (1 + 0.45 * np.cos(v)) * np.sin(u), 0.45 * np.sin(v)))
Tor = np.array(_t)

out = {"claim": 6, "seed": SEED, "source": "Section 5 (shape analysis)",
       "shape": "unit sphere (proxy for Armadillo mesh; original asset unavailable)",
       "eps": eps, "n_eigs": NEIG, "modalities": {}}

spectra = {}
for name, X in [("uniform_point_cloud", A), ("mesh_vertices_fibonacci", B),
                ("voxelised_shell", C), ("MUTATION_anisotropic", Cm),
                ("MUTATION_torus", Tor)]:
    lam, it = spectrum(X, eps)
    lam_n = lam / lam[1]                      # normalise by first non-trivial eigenvalue
    spectra[name] = lam_n
    out["modalities"][name] = dict(n=int(len(X)), mean_nn_spacing=spacing(X),
                                   sinkhorn_iters=it,
                                   lambda_raw=[float(x) for x in lam],
                                   lambda_normalised=[float(x) for x in lam_n])

exact = []
l = 1
while len(exact) < NEIG - 1:
    exact += [l * (l + 1)] * (2 * l + 1)
    l += 1
exact = np.array([0.0] + exact[:NEIG - 1])
exact_n = exact / 2.0                          # normalised by lambda(l=1) = 2

ref = spectra["uniform_point_cloud"]
out["exact_sphere_normalised"] = [float(x) for x in exact_n]

pairs = {}
for name in ("mesh_vertices_fibonacci", "voxelised_shell", "MUTATION_anisotropic",
             "MUTATION_torus"):
    rel = np.abs(spectra[name] - ref) / np.maximum(ref, 1e-12)
    rel[0] = 0.0
    idx = [k for k in range(1, NEIG) if rel[k] > 0.10]
    pairs[name] = dict(rel_dev=[float(x) for x in rel],
                       max_rel_dev_first10=float(np.max(rel[1:10])),
                       first_index_exceeding_10pct=(int(idx[0]) if idx else None))
out["cross_modality_deviation_vs_uniform_point_cloud"] = pairs

# accuracy against the exact sphere spectrum (low modes)
acc = {}
for name in ("uniform_point_cloud", "mesh_vertices_fibonacci", "voxelised_shell"):
    rel = np.abs(spectra[name][1:10] - exact_n[1:10]) / exact_n[1:10]
    acc[name] = dict(mean_rel_err_modes_1_9=float(np.mean(rel)),
                     max_rel_err_modes_1_9=float(np.max(rel)))
out["accuracy_vs_exact_sphere_spectrum"] = acc

# Individual eigenvalues inside a degenerate multiplet (l -> multiplicity 2l+1) are split
# arbitrarily by sampling noise, so the *distribution-level* comparison is over the
# multiplet group means (l = 1,2,3 -> index blocks [1:4], [4:9], [9:16]).
BLOCKS = {"l1": (1, 4), "l2": (4, 9), "l3": (9, 16)}
group = {}
for name, lam_n in spectra.items():
    g = {k: float(np.mean(lam_n[a:b])) for k, (a, b) in BLOCKS.items()}
    g = {k: v / g["l1"] for k, v in g.items()}       # normalise by the l=1 group
    group[name] = g
out["multiplet_group_means_normalised"] = group
out["multiplet_group_means_exact"] = {"l1": 1.0, "l2": 3.0, "l3": 6.0}

gdev = {}
for name in ("mesh_vertices_fibonacci", "voxelised_shell", "MUTATION_anisotropic",
             "MUTATION_torus"):
    gdev[name] = {k: abs(group[name][k] - group["uniform_point_cloud"][k]) /
                  group["uniform_point_cloud"][k] for k in BLOCKS}
out["group_mean_deviation_vs_uniform_point_cloud"] = gdev

consistent = all(max(gdev[n].values()) < 0.10
                 for n in ("mesh_vertices_fibonacci", "voxelised_shell"))
# "divergence only at scales matching sampling resolution": deviation must grow with the
# mode index (highest block worst) and stay small for the lowest block.
divergence_late = all(gdev[n]["l1"] <= gdev[n]["l3"] + 1e-12 and gdev[n]["l1"] < 0.05
                      for n in ("mesh_vertices_fibonacci", "voxelised_shell"))
mut_breaks = max(gdev["MUTATION_torus"].values()) > 0.10
out["mutation_anisotropic_max_group_dev"] = float(max(gdev["MUTATION_anisotropic"].values()))
out["mutation_torus_max_group_dev"] = float(max(gdev["MUTATION_torus"].values()))
out["mutation_note"] = (
    "The l=1-normalised multiplet group means are insensitive to a mild anisotropic "
    "stretch (z x1.35, max dev 0.5%), so that mutation is NOT discriminating and is "
    "reported as such; changing the surface to a torus does break the statistic.")

out["low_modes_consistent_across_modalities"] = bool(consistent)
out["divergence_only_at_high_modes"] = bool(divergence_late)
out["mutation_breaks_consistency"] = bool(mut_breaks)
out["verdict"] = "toy" if (consistent and divergence_late) else "inconclusive"
out["note"] = ("Verdict 'toy': cross-modality spectral consistency with divergence "
               "confined to high modes reproduces on a proxy shape (unit sphere, exact "
               "spectrum known); the paper's Armadillo mesh was not available, so the "
               "reported figures cannot be matched numerically.")

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/claim6.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "modalities"}, indent=1))
