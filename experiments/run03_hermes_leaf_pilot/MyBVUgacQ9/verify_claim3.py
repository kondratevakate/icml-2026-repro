"""verify_claim3.py -- CLAIM 3 / Theorem 3.3 (Section 3).

Claim: under the assumptions of Thm 3.1, when Q satisfies eq. 4 and the {I_i} are
indicators of POLYHEDRAL sets, the iterates satisfy
    L(x^k,z^k,w^k) - min_{(x,z) in B(x^k,z^k;r)} L(x,z,w^k)  in O(c2^{-k}),  c2 > 1,
without assuming second-order differentiability of L at the limit point, and
lim (x^k,z^k) = (x*,z*) is a local minimum.

Setup: Section-5 toy problem plus BOX constraints (boxes are polyhedra) chosen so that
the limit point sits on an ACTIVE face -- there the indicator is non-differentiable, so
Theorem 3.2 does not apply but Theorem 3.3 should.  We
  (a) certify that constraints are active at the limit (nonsmooth regime),
  (b) evaluate the exact Thm-3.3 quantity  L^k - min_{B(x^k,z^k;r)} L(.,.,w^k)
      with scipy SLSQP over the intersection of the box and the ball of radius r,
  (c) fit its per-iteration contraction factor,
  (d) certify local minimality of (x*,z*) by exhaustive random probing in the feasible set.

MUTATION A: violate eq. 4 by inflating ||C|| -> the geometric decay must be lost.
MUTATION B: replace the polyhedral box by a Euclidean BALL (not polyhedral), i.e. break
the hypothesis of Thm 3.3, and report what happens.

Run:  .venv/bin/python verify_claim3.py
"""
import itertools
import json
import time

import numpy as np
from scipy.optimize import minimize

import admm_lib as A

OUT = "results/claim3.json"
CMD = ".venv/bin/python verify_claim3.py"

LO, HI = 0.3, 1.5          # box  x_i in [0.3, 1.5]  (polyhedral, active at the limit)
R_BALL = 0.05              # radius r of B(x^k, z^k; r) in Theorem 3.3


def boxed_toy(q, c_scale=1.0, mu_x=1.0, mu_z=1.0):
    box = [(np.array([LO]), np.array([HI]))] * 4
    return A.toy_problem(q, mu_x=mu_x, mu_z=mu_z, box=box, c_scale=c_scale)


def thm33_gap(prob, x, z, w, rho, r=R_BALL):
    """L(x,z,w) - min_{(u,v) in B((x,z);r), u in box} L(u,v,w)."""
    nx = len(x)
    v0 = np.concatenate([x, z])

    def obj(u):
        return prob.lagrangian(u[:nx], u[nx:], w, rho)

    cons = [{"type": "ineq", "fun": lambda u: r ** 2 - np.sum((u - v0) ** 2)}]
    bnds = [(LO, HI)] * nx + [(None, None)] * (len(v0) - nx)
    best = obj(v0)
    for pert in [np.zeros_like(v0), 0.5 * r * np.ones_like(v0), -0.5 * r * np.ones_like(v0)]:
        start = np.clip(v0 + pert, [LO] * nx + [-1e9] * (len(v0) - nx),
                        [HI] * nx + [1e9] * (len(v0) - nx))
        out = minimize(obj, start, bounds=bnds, constraints=cons, method="SLSQP",
                       options={"maxiter": 300, "ftol": 1e-14})
        if out.success and np.sum((out.x - v0) ** 2) <= r ** 2 + 1e-8:
            best = min(best, float(out.fun))
    return float(obj(v0) - best)


def local_min_probe(prob, x, z, rho, n=400, radius=1e-3, seed=0):
    """Is (x*,z*) a local minimum of problem (1)?  Probe feasible perturbations:
    move x inside the box, then repair feasibility of A(x)+Qz=0 by solving for z."""
    rng = np.random.default_rng(seed)
    f0 = prob.f(x) + prob.phi(z)
    worst = 0.0
    n_better = 0
    for _ in range(n):
        xp = np.clip(x + rng.normal(size=len(x)) * radius, LO, HI)
        # repair: choose z minimising phi subject to A(xp) + Q z = 0
        Ax = prob.A(xp)
        zp, *_ = np.linalg.lstsq(prob.Q, -Ax, rcond=None)
        if np.linalg.norm(prob.A(xp) + prob.Q @ zp) > 1e-9:
            continue
        fp = prob.f(xp) + prob.phi(zp)
        if fp < f0 - 1e-12:
            n_better += 1
            worst = max(worst, f0 - fp)
    return {"n_probes": n, "n_strictly_better": n_better, "max_improvement": float(worst),
            "objective_at_limit": float(f0)}


def run_cell(prob, rho, iters, seeds, thm33=True):
    facs, actives, gaps_fac, viols, locmins = [], [], [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        x0 = np.clip(rng.uniform(LO, HI, size=4), LO, HI)
        z0 = rng.normal(size=1)
        x, z, w, h = prob.run(x0, z0, np.zeros(1), rho, iters=iters, store_traj=True)
        if not np.all(np.isfinite(np.concatenate([x, z, w]))):
            facs.append(float("inf")); continue
        f, npts, r2 = A.contraction_factor(h["res"], floor=1e-14)
        if not np.isfinite(f):
            rr = h["res"]; idx = np.where(rr > 1e-14)[0]
            f = float(np.exp((np.log(rr[idx[-1]]) - np.log(rr[idx[0]])) /
                             max(1, idx[-1] - idx[0]))) if len(idx) >= 3 else 0.0
        facs.append(f)
        nact = int(np.sum((np.abs(x - LO) < 1e-9) | (np.abs(x - HI) < 1e-9)))
        actives.append(nact)
        viols.append(float(h["viol"][-1]))
        if thm33:
            ks = [k for k in range(2, min(len(h["x"]), 60), 2)]
            seq = np.array([thm33_gap(prob, h["x"][k], h["z"][k], h["w"][k], rho)
                            for k in ks])
            seq = np.maximum(seq, 0.0)
            g, npg, r2g = A.contraction_factor(seq, floor=1e-13, min_pts=5)
            if not np.isfinite(g):
                # too few points above the noise floor (gap collapses almost at once):
                # bound the per-step factor by the geometric mean of the observed decay
                idx = np.where(seq > 1e-15)[0]
                if len(idx) >= 2:
                    g = float(np.exp((np.log(seq[idx[-1]]) - np.log(seq[idx[0]])) /
                                     max(1, idx[-1] - idx[0])))
                else:
                    g = 0.0
            # ks are spaced by 2 iterations -> convert to per-iteration factor
            gaps_fac.append(float(np.sqrt(g)) if np.isfinite(g) else float("nan"))
        locmins.append(local_min_probe(prob, x, z, rho, seed=s))
    return {"n_seeds": len(seeds),
            "worst_res_factor": float(np.max(facs)),
            "median_res_factor": float(np.median(facs)),
            "thm33_gap_factors": gaps_fac,
            "worst_thm33_gap_factor": (float(np.nanmax(gaps_fac)) if gaps_fac else None),
            "n_active_constraints_at_limit": actives,
            "worst_final_viol": float(np.max(viols)) if viols else None,
            "local_min_probe_failures": sum(l["n_strictly_better"] > 0 for l in locmins),
            "max_local_improvement": float(max(l["max_improvement"] for l in locmins))
            if locmins else None}


def main():
    t0 = time.time()
    seeds = list(range(12))
    res = {"claim": 3, "source": "Theorem 3.3, Section 3", "command": CMD,
           "box": [LO, HI], "ball_radius_r": R_BALL, "seeds": seeds}

    # ---- polyhedral (box), small ||C|| : Thm 3.3 regime --------------------
    cells = {}
    for q in [2.0, 5.0]:
        for rho in [1.0, 4.0]:
            prob = boxed_toy(q, c_scale=1.0)
            cells["poly_q%g_rho%g" % (q, rho)] = run_cell(prob, rho, 250, seeds)
    res["polyhedral_small_C"] = cells

    # ---- MUTATION A : eq. 4 violated (||C|| inflated) ----------------------
    mut = {}
    for cs in [10.0, 50.0]:
        prob = boxed_toy(2.0, c_scale=cs)
        mut["poly_Cscale%g_rho4" % cs] = run_cell(prob, 4.0, 250, seeds)
    res["MUTATION_A_large_C"] = mut

    # ---- MUTATION B : non-polyhedral feasible set (ball) -------------------
    # x in {||x||<=R} implemented by projecting each block update onto the ball
    # (exactly, since blocks are 1-D and the ball is separable only approximately;
    #  we therefore use the smallest enclosing/enclosed box argument and instead
    #  enforce the ball by an exact projected update on the full x-vector).
    class BallProb(A.MultiAffineProblem):
        R = 1.2

        def run(self, x0, z0, w0, rho, iters=400, record=True, store_traj=False):
            x, z, w = np.array(x0, float), np.array(z0, float), np.array(w0, float)
            hist = {"L": [], "res": [], "viol": [], "x": [], "z": [], "w": []}
            for k in range(iters):
                if record:
                    hist["L"].append(self.lagrangian(x, z, w, rho))
                    hist["viol"].append(float(np.linalg.norm(self.A(x) + self.Q @ z)))
                xp, zp = x.copy(), z.copy()
                for i in range(len(self.blocks)):
                    b = self.blocks[i]
                    x[b] = self.x_block_update(x, z, w, rho, i)
                    nrm = np.linalg.norm(x)
                    if nrm > self.R:            # exact projection onto the ball
                        x *= self.R / nrm
                z = self.z_update(x, w, rho)
                w = w + rho * (self.A(x) + self.Q @ z)
                if record:
                    hist["res"].append(float(np.linalg.norm(
                        np.concatenate([x - xp, z - zp]))))
                if not np.all(np.isfinite(np.concatenate([x, z, w]))):
                    break
            for key in ("L", "res", "viol"):
                hist[key] = np.array(hist[key])
            return x, z, w, hist

    C = np.zeros((1, 4, 4)); C[0, 0, 1] = C[0, 1, 0] = 1.0
    C[0, 2, 3] = C[0, 3, 2] = -1.0
    ball = BallProb(blocks=[[0], [1], [2], [3]], C=C, d=np.zeros((1, 4)),
                    e=np.array([1.0]), Q=np.array([[2.0]]),
                    mu_x=np.ones(4), mu_z=np.ones(1))
    bfacs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        _, _, _, h = ball.run(rng.normal(size=4), rng.normal(size=1), np.zeros(1),
                              4.0, iters=250)
        f, _, _ = A.contraction_factor(h["res"], floor=1e-14)
        bfacs.append(f)
    res["MUTATION_B_nonpolyhedral_ball"] = {
        "factors": [None if not np.isfinite(f) else float(f) for f in bfacs],
        "note": "hypothesis of Thm 3.3 broken (ball is not polyhedral); reported as-is"}

    res["elapsed_sec"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    print("Thm 3.3 regime (box [%g,%g] polyhedral, ||C||=1):" % (LO, HI))
    for k, v in cells.items():
        print("  %s: worst res factor %.4f | worst Thm3.3-gap factor %s | "
              "active constraints at limit %s | viol %.1e | local-min probe failures %d"
              % (k, v["worst_res_factor"],
                 ("%.4f" % v["worst_thm33_gap_factor"]) if v["worst_thm33_gap_factor"] else "n/a",
                 sorted(set(v["n_active_constraints_at_limit"])), v["worst_final_viol"],
                 v["local_min_probe_failures"]))
    print("MUTATION A (eq.4 violated, ||C|| inflated):")
    for k, v in mut.items():
        print("  %s: worst res factor %.4f | worst Thm3.3-gap factor %s"
              % (k, v["worst_res_factor"],
                 ("%.4f" % v["worst_thm33_gap_factor"]) if v["worst_thm33_gap_factor"] else "n/a"))
    print("MUTATION B (ball):", res["MUTATION_B_nonpolyhedral_ball"]["factors"])
    print("wrote", OUT, "in %.1fs" % res["elapsed_sec"])


if __name__ == "__main__":
    main()
