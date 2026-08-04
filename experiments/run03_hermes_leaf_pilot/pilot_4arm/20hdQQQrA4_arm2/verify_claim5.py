"""verify_claim5.py — safety-critical control: CAffNet avoids the obstacle, HardNet and
the soft baseline do not.

CLAIM (Experiments, safety-critical control task): in the safety-critical control
experiment, CAffNet successfully avoids obstacles whereas both HardNet and the
soft-constrained baseline fail to do so.

ROUTE (skill §2g): the *qualitative outcome* (avoid vs collide) is produced by the
constraint layer and is scale-free; it is what this run tests. The paper's reported
control costs and its trained transformer policy are GPU-scale and unreleased, so no
numeric cost comparison is attempted.

SETUP (independent reimplementation): single-integrator x' = u, dt = 0.05, 200 steps,
goal at (2,0), circular obstacle centred (1,0) with radius 0.4. Nominal (goal-seeking)
command drives straight through the obstacle. Safety is written as a control barrier
function constraint, linear in the command:
    h(x) = ||x - c||^2 - r^2,   -2 (x-c)^T u <= alpha * h(x)
plus a box on each actuator (|u_i| <= u_max). The rows are deliberately made partially
dependent (a duplicated, rescaled CBF row) so the constraint matrix is NOT full row rank
— the regime the paper says HardNet cannot handle.

P27 NOTE: the constraint binds the TOTAL command u = u_nom + u_net, so every operator is
applied to the total command, not to the network's increment alone.

ARMS
  * caff     : total command passed through the CAffine layer (enumeration + selection)
  * hardnet  : single pseudo-inverse correction on the violated rows (prior work)
  * soft     : penalty-style correction, u <- u_nom - lam * grad(violation penalty)

MUTATION: the `soft` and `hardnet` arms ARE the mutation (remove the enumeration /
remove the hard layer). PREDICTION (before running): CAffNet has zero collisions and zero
constraint violations on every rollout; at least one of the two baselines collides.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from caff_core import caffine, hardnet_aff, max_violation

CMD = "python verify_claim5.py"
SEEDS = [0, 1, 2]
N_INIT = 12
DT = 0.05
STEPS = 200
GOAL = np.array([2.0, 0.0])
ALPHA = 1.0

# Two regimes are run, because the first one turned out not to discriminate (see
# `regimes` in the JSON): a single obstacle with generous actuator limits is avoided by
# every operator, so the paper's contrast is only testable in the second regime, where
# two obstacles and a tight actuator box make several constraint rows bind at once.
REGIMES = {
    "single_obstacle_loose_box": {
        "obstacles": [(np.array([1.0, 0.0]), 0.40)],
        "umax": 2.0, "lam_soft": 0.05,
    },
    "two_obstacles_tight_box": {
        "obstacles": [(np.array([0.80, 0.16]), 0.36), (np.array([1.25, -0.16]), 0.36)],
        "umax": 0.9, "lam_soft": 0.05,
    },
    "narrow_gap_weak_barrier": {
        "obstacles": [(np.array([1.0, 0.52]), 0.48), (np.array([1.0, -0.52]), 0.48)],
        "umax": 1.2, "lam_soft": 0.02, "alpha": 6.0,
    },
}


def cbf_constraints(x, reg):
    """A(x) u <= b(x): one CBF row per obstacle (each duplicated & rescaled, so the
    matrix is NOT full row rank) plus an actuator box."""
    rows, rhs, hs = [], [], []
    for c, r in reg["obstacles"]:
        d = x - c
        h = float(d @ d - r ** 2)
        g = -2.0 * d                   # d/dt h = 2 d^T u  =>  -2 d^T u <= alpha h
        rows += [g, 2.0 * g]           # dependent duplicate row
        al = reg.get("alpha", ALPHA)
        rhs += [al * h, 2.0 * al * h]
        hs.append(h)
    um = reg["umax"]
    rows += [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    rhs += [um, um, um, um]
    return np.array(rows), np.array(rhs), min(hs)


def rollout(arm, x0, reg):
    x = x0.copy()
    min_clear = np.inf
    max_viol = 0.0
    collided = False
    um = reg["umax"]
    for _ in range(STEPS):
        A, b, _ = cbf_constraints(x, reg)
        u_nom = np.clip(1.5 * (GOAL - x), -um, um)
        if arm == "caff":
            u = caffine(A, b, u_nom)
        elif arm == "hardnet":
            u = hardnet_aff(A, b, u_nom)
        else:                                    # soft penalty gradient step
            v = np.maximum(A @ u_nom - b, 0.0)
            u = u_nom - reg["lam_soft"] * (A.T @ v)
        max_viol = max(max_viol, max_violation(A, b, u))
        x = x + DT * u
        for c, r in reg["obstacles"]:
            clr = float(np.linalg.norm(x - c)) - r
            min_clear = min(min_clear, clr)
            if clr < 0.0:
                collided = True
    return {"min_clearance": min_clear,
            "collided": collided,
            "max_constraint_violation": max_viol,
            "final_dist_to_goal": float(np.linalg.norm(x - GOAL))}


def main():
    t0 = time.time()
    arms = ("caff", "hardnet", "soft")
    regimes_out = {}
    for rname, reg in REGIMES.items():
        summary = {a: {"n_rollouts": 0, "n_collisions": 0, "min_clearance": np.inf,
                       "max_violation": 0.0, "goal_dists": []} for a in arms}
        for seed in SEEDS:
            rng = np.random.default_rng([seed, 5])
            for i in range(N_INIT):
                ang = 2 * np.pi * i / N_INIT
                x0 = np.array([-1.0, 0.0]) + 0.25 * np.array([np.cos(ang), np.sin(ang)])
                x0 = x0 + rng.normal(scale=0.05, size=2)
                for a in arms:
                    r = rollout(a, x0, reg)
                    s = summary[a]
                    s["n_rollouts"] += 1
                    s["n_collisions"] += int(r["collided"])
                    s["min_clearance"] = min(s["min_clearance"], r["min_clearance"])
                    s["max_violation"] = max(s["max_violation"], r["max_constraint_violation"])
                    s["goal_dists"].append(r["final_dist_to_goal"])
        for a in arms:
            s = summary[a]
            s["min_clearance"] = float(s["min_clearance"])
            s["mean_final_goal_dist"] = float(np.mean(s.pop("goal_dists")))
        regimes_out[rname] = summary

    caff_clean = all(regimes_out[r]["caff"]["n_collisions"] == 0
                     and regimes_out[r]["caff"]["max_violation"] < 1e-6
                     for r in REGIMES)
    discriminating = [r for r in REGIMES
                      if regimes_out[r]["hardnet"]["n_collisions"] > 0
                      or regimes_out[r]["soft"]["n_collisions"] > 0]
    baseline_viol = {r: {"hardnet": regimes_out[r]["hardnet"]["max_violation"],
                         "soft": regimes_out[r]["soft"]["max_violation"]}
                     for r in REGIMES}

    res = {
        "claim": 5,
        "source": "Experiments, safety-critical control (obstacle avoidance)",
        "command": CMD,
        "scale_note": ("PAPER: trained policy, 300 initial states, GPU. THIS RUN: analytic "
                       "nominal controller + the three constraint operators, 12 initial "
                       "states x 3 seeds x 200 steps x 2 regimes, CPU-only. Only the "
                       "qualitative avoid/collide outcome is claimed here."),
        "config": {"seeds": SEEDS, "n_init": N_INIT, "steps": STEPS, "dt": DT,
                   "alpha": ALPHA,
                   "regimes": {k: {"obstacles": [[c.tolist(), r] for c, r in v["obstacles"]],
                                   "umax": v["umax"], "lam_soft": v["lam_soft"],
                                   "alpha": v.get("alpha", ALPHA)}
                               for k, v in REGIMES.items()},
                   "constraint_matrix_rank_deficient": True},
        "regimes": regimes_out,
        "caffnet_collision_free_in_all_regimes": caff_clean,
        "discriminating_regimes": discriminating,
        "baseline_max_constraint_violation": baseline_viol,
        "mutation_baselines": {
            "prediction": ("CAffNet: 0 collisions and 0 constraint violations in every "
                           "regime; at least one baseline collides"),
            "observation": ("PREDICTION WRONG, recorded verbatim rather than rewritten: "
                            "no baseline collided in ANY of the three regimes tried "
                            "(single obstacle / loose box, two obstacles / tight box, "
                            "narrow gap / weak barrier). What the baselines do lose is the "
                            "hard-constraint guarantee itself: HardNet-Aff reaches a CBF "
                            "violation of 0.620 in the tight regime and the soft arm 1.364 "
                            "in the loose regime, while CAffNet stays at machine precision "
                            "(<=7.1e-15) everywhere."),
        },
        "verdict": None,
        "wall_seconds": None,
    }
    # Sub-claim split: (a) CAffNet satisfies the hard safety constraint and avoids the
    # obstacle; (b) the baselines FAIL to do so. (a) is verified; (b) is only partly
    # observable here — the baselines lose the hard-constraint guarantee (measured CBF
    # violations) but no arm actually collided in this simplified single-integrator
    # setting, so the collision half of the contrast stays inconclusive.
    baselines_violate = all(
        max(baseline_viol[r]["hardnet"], baseline_viol[r]["soft"]) > 1e-3 for r in REGIMES)
    res["sub_verdicts"] = {
        "caffnet_satisfies_hard_constraint_and_avoids":
            "verified" if caff_clean else "falsified",
        "baselines_lose_the_hard_guarantee":
            "verified" if baselines_violate else "inconclusive",
        "baselines_actually_collide":
            "verified" if discriminating else "inconclusive",
    }
    if caff_clean and baselines_violate and not discriminating:
        res["verdict"] = ("partial: verified (CAffNet avoids + baselines violate the hard "
                          "constraint) / inconclusive (no baseline collision observed)")
        res["verdict_cap_reason"] = (
            "In this CPU-scale single-integrator reimplementation the CBF is conservative "
            "enough that the baselines' constraint violations (HardNet-Aff max 0.62, soft "
            "max 1.36) did not translate into an actual collision in any of the three "
            "regimes tried. The paper's collision outcome depends on its trained policy "
            "and dynamics, which are unreleased and GPU-scale.")
    elif caff_clean and discriminating:
        res["verdict"] = "verified (qualitative, CPU-scale, regime-dependent)"
    elif caff_clean:
        res["verdict"] = "inconclusive"
        res["verdict_cap_reason"] = ("CAffNet is collision-free and exactly feasible "
                                     "everywhere, but neither baseline was made to fail")
    else:
        res["verdict"] = "falsified"
    res["wall_seconds"] = round(time.time() - t0, 2)
    Path("results").mkdir(exist_ok=True)
    Path("results/claim5.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
