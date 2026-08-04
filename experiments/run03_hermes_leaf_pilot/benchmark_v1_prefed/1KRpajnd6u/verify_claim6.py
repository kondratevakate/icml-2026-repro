"""verify_claim6.py — D-head: exact conservation, but only *empirical* dual-bound control.

Claim under test (Sec 3.4 / Eq. 3-4): the D-head enforces upper+lower bounds via the dual
consistency loss (DCL) rather than a strict architectural guarantee, achieving empirically
near-zero violations rather than the exact guarantees proven for L-head/U-head.

Three things are measured:
  (a) the averaged update (Eq. 3) is still exactly conservative;
  (b) with untrained / disagreeing branches, bound violations DO occur -> no strict guarantee;
  (c) MECHANISM TEST: actually minimizing L_DCL (Eq. 4) by gradient descent on the head logits
      drives the violation rate to (near) zero -- i.e. it is DCL, not the architecture, that
      enforces the dual bound. MUTATION: flip the sign of the DCL objective (maximize branch
      disagreement) for the same number of steps -> violations must NOT vanish but grow.
      Additionally the rank correlation between sqrt(DCL) and the violation magnitude over an
      ensemble of random states is reported.
  (d) analytic bound: any violation is at most 0.5*|du_out - du_in|, so violations vanish iff
      the branches agree.

Run: .venv/bin/python verify_claim6.py
Writes results/claim6.json
"""
import json, sys
import numpy as np
import torch
from fluxnet_core import stencil, d_head_update

OUT = "results/claim6.json"
res = {"command": ".venv/bin/python verify_claim6.py", "claim": 6}
L, UMAX = 0.0, 1.0

# ---------------- (a) conservation of the averaged update ----------------
def cons(shape, radius, seed, steps=10):
    rng = np.random.default_rng(seed)
    ndim = len(shape); offs = stencil(radius, ndim); K = len(offs)
    u = rng.random(shape); worst = 0.0
    for _ in range(steps):
        s0 = u.sum()
        un, _, _ = d_head_update(u, L, UMAX, rng.normal(0, 2, shape),
                                 rng.normal(0, 2, (K,) + shape), rng.normal(0, 2, shape),
                                 rng.normal(0, 2, (K,) + shape), offs)
        worst = max(worst, abs(un.sum() - s0) / s0)
        u = np.clip(un, L, UMAX)
    return worst

res["d_head_max_conservation_rel_err_float64"] = max(
    cons(sh, r, s) for sh in [(64,), (32, 32)] for r in [1, 2] for s in range(10))

# ---------------- torch D-head (differentiable, identical math) ----------------
def torch_offsets(radius, ndim):
    return stencil(radius, ndim)

def d_head_torch(u, al, pil, bl, rhl, offs):
    ndim = u.dim()
    axes = tuple(range(ndim))
    a = u - L
    alpha = torch.sigmoid(al)
    pi = torch.softmax(pil, dim=0)
    Fout = (a * alpha).unsqueeze(0) * pi
    inflow = sum(torch.roll(Fout[k], shifts=d, dims=axes) for k, d in enumerate(offs))
    du_out = -Fout.sum(0) + inflow

    b = UMAX - u
    beta = torch.sigmoid(bl)
    rho = torch.softmax(rhl, dim=0)
    G = (b * beta).unsqueeze(0) * rho
    outflow = sum(torch.roll(G[k], shifts=tuple(-x for x in d), dims=axes)
                  for k, d in enumerate(offs))
    du_in = G.sum(0) - outflow
    return u + 0.5 * (du_out + du_in), du_out, du_in


def optimize(seed, shape=(32, 32), radius=2, steps=200, objective="dcl"):
    """Optimize the head logits; return (dcl, lb_rate%, ub_rate%, max_mag) before and after."""
    g = torch.Generator().manual_seed(seed)
    offs = torch_offsets(radius, len(shape)); K = len(offs)
    u = torch.rand(shape, generator=g, dtype=torch.float64)
    p = [torch.randn(shape, generator=g, dtype=torch.float64) * 3.0,
         torch.randn((K,) + shape, generator=g, dtype=torch.float64) * 3.0,
         torch.randn(shape, generator=g, dtype=torch.float64) * 3.0,
         torch.randn((K,) + shape, generator=g, dtype=torch.float64) * 3.0]
    for t in p:
        t.requires_grad_(True)

    def metrics():
        with torch.no_grad():
            un, do, di = d_head_torch(u, *p, offs)
            dcl = float(((do - di) ** 2).mean())
            lb = float((un < L - 1e-12).double().mean() * 100)
            ub = float((un > UMAX + 1e-12).double().mean() * 100)
            mag = float(torch.clamp(torch.maximum(L - un, un - UMAX), min=0).max())
        return dict(dcl=dcl, lb_rate=lb, ub_rate=ub, max_mag=mag)

    before = metrics()
    opt = torch.optim.Adam(p, lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        _, do, di = d_head_torch(u, *p, offs)
        dcl = ((do - di) ** 2).mean()               # Eq. 4
        loss = dcl if objective == "dcl" else -dcl  # mutation: maximize disagreement
        loss.backward()
        opt.step()
    return before, metrics()


seeds = list(range(6))
dcl_runs = [optimize(s, objective="dcl") for s in seeds]
ctl_runs = [optimize(s, objective="anti_dcl") for s in seeds]

def agg(runs, key):
    return {k: float(np.mean([r[key][k] for r in runs])) for k in
            ("dcl", "lb_rate", "ub_rate", "max_mag")}

res["untrained_branches"] = agg(dcl_runs, 0)
res["untrained_branches"]["n_seeds"] = len(seeds)
res["untrained_branches"]["n_seeds_with_violation"] = int(
    sum(1 for r in dcl_runs if r[0]["lb_rate"] + r[0]["ub_rate"] > 0))
res["after_DCL_minimization"] = agg(dcl_runs, 1)
res["after_DCL_minimization"]["n_seeds_zero_violation"] = int(
    sum(1 for r in dcl_runs if r[1]["lb_rate"] + r[1]["ub_rate"] == 0.0))
res["mutation_maximize_DCL"] = agg(ctl_runs, 1)
res["mutation_maximize_DCL"]["n_seeds_zero_violation"] = int(
    sum(1 for r in ctl_runs if r[1]["lb_rate"] + r[1]["ub_rate"] == 0.0))
res["optimizer"] = {"adam_lr": 0.05, "steps": 200, "grid": "32x32", "radius": 2,
                    "dtype": "float64"}

# ---------------- (d) analytic bound: violation <= 0.5*|du_out - du_in| ----------------
def check_analytic(seed):
    rng = np.random.default_rng(seed); shape = (32, 32); offs = stencil(2, 2); K = len(offs)
    u = rng.random(shape)
    un, du_out, du_in = d_head_update(u, L, UMAX, rng.normal(0, 5, shape),
                                      rng.normal(0, 5, (K,) + shape), rng.normal(0, 5, shape),
                                      rng.normal(0, 5, (K,) + shape), offs)
    viol = np.maximum(np.maximum(L - un, un - UMAX), 0.0)
    return bool(np.all(viol <= 0.5 * np.abs(du_out - du_in) + 1e-12)), float(viol.max())

ana = [check_analytic(s) for s in range(20)]
res["analytic_bound_violation_le_half_branch_gap"] = {
    "holds_all_20_seeds": all(a[0] for a in ana),
    "max_observed_violation": max(a[1] for a in ana)}

# ---------------- (e) DCL <-> violation-magnitude correlation over an ensemble -------------
def dcl_and_violation(seed, scale):
    rng = np.random.default_rng(seed); shape = (32, 32); offs = stencil(2, 2); K = len(offs)
    u = rng.random(shape)
    un, do, di = d_head_update(u, L, UMAX, rng.normal(0, scale, shape),
                               rng.normal(0, scale, (K,) + shape), rng.normal(0, scale, shape),
                               rng.normal(0, scale, (K,) + shape), offs)
    v = float(np.maximum(np.maximum(L - un, un - UMAX), 0.0).max())
    return float(np.sqrt(np.mean((do - di) ** 2))), v

pairs = [dcl_and_violation(s, sc) for sc in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0) for s in range(15)]
x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
from scipy.stats import spearmanr
rho, pval = spearmanr(x, y)
res["dcl_vs_violation_spearman"] = {"rho": float(rho), "p": float(pval), "n": len(pairs),
                                    "n_zero_violation": int((y == 0).sum())}

res["verdict"] = ("verified" if
                  res["d_head_max_conservation_rel_err_float64"] < 1e-13
                  and res["untrained_branches"]["n_seeds_with_violation"] == len(seeds)
                  and res["after_DCL_minimization"]["dcl"] < 0.01 * res["untrained_branches"]["dcl"]
                  and res["after_DCL_minimization"]["max_mag"] < 0.05 * res["untrained_branches"]["max_mag"]
                  and res["mutation_maximize_DCL"]["n_seeds_zero_violation"] == 0
                  and res["dcl_vs_violation_spearman"]["rho"] > 0.7
                  and res["analytic_bound_violation_le_half_branch_gap"]["holds_all_20_seeds"]
                  else "falsified")
res["env"] = {"python": sys.version.split()[0], "numpy": np.__version__,
              "torch": torch.__version__}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
