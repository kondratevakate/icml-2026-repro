"""verify_claim5.py — Sec. 4.3 safety-critical control: "CAffNet avoids obstacles whereas
HardNet and the soft-constrained baseline fail".

WHAT IS AND IS NOT REPRODUCIBLE HERE (see logbook 'Evidence boundary'):
The paper (App. D.3) gives the obstacles, the input/state boxes, the PID nominal controller
gains, and x0 = [-4.5, 0, 0.5], but it does NOT specify: the reference trajectory, dt,
the horizon, the training loss, or — crucially — the *affine* per-step safety constraint
A(x)u <= b(x) that encodes obstacle avoidance (obstacle avoidance itself is a NON-convex
complement-of-polytope condition, so some hyperplane-selection scheme must be supplied).
Therefore this script does NOT reproduce the trained policies of Table 4. It tests the
*mechanism-level* sub-question that is well posed from the paper's own data:

  given the same nominal PID controller and the same per-step affine safety constraints
  (a discriminating-hyperplane / CBF-style constraint per obstacle, m = 3 + 4 = 7 > n_out = 2,
  which is exactly the regime the paper claims HardNet cannot handle),
  does the CAffNet projection keep the closed loop collision-free while the soft baseline
  (unprojected, clipped nominal input) and a HardNet-style single pseudo-inverse
  correction do not?

MUTATION: disable the combination search (kmax = 1 only / no projection) and check the
collision count changes as predicted.
"""
import json, time
import numpy as np
from caffnet_core import caffnet, hardnet_like

OUT = "results/claim5.json"

# ---- obstacles (App. D.3) ----
A1 = np.array([[0.4472, -0.8944], [0.7071, 0.7071], [-0.2425, 0.9701],
               [-0.7071, -0.7071], [-0.8944, -0.4472]])
b1 = np.array([-0.2184, -0.5303, 0.6219, 1.1667, 1.4368])
A2 = np.array([[-0.9685, 0.2489], [0.9417, 0.3363], [-0.3714, 0.9285],
               [0.3714, 0.9285], [-0.9417, -0.3363], [-0.2976, -0.9547]])
b2 = np.array([1.2755, -1.7670, -0.8511, -2.0249, 2.3274, 2.6868])
A3 = np.array([[-0.9191, 0.3939], [0.8944, 0.4472], [0.9703, -0.2419],
               [-0.8701, -0.4930], [0.0000, -1.0000]])
b3 = np.array([2.9916, -1.9975, -2.5305, 2.4854, 0.1000])
OBS = [(A1, b1), (A2, b2), (A3, b3)]

# ---- input box (App. D.3): A_u u <= b_u ----
AU = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
BU = np.array([1.0, 0.01, 0.5, 0.5])

KP = np.diag([0.01, 0.2, 0.0]); KI = np.diag([0.05, 0.005, 0.0]); KD = np.diag([0.0, 0.01, 0.0])
SEL = np.array([[1.0, 0, 0], [0, 1.0, 1.0]])
DT, STEPS = 0.05, 1200
MARGIN = 0.05


def inside(p, A, b, tol=0.0):
    return bool(np.all(A @ p <= b + tol))


def safety_rows(p, theta):
    """One affine-in-u row per obstacle: keep the currently most-satisfied separating
    hyperplane of that obstacle satisfied at the next Euler step (discriminating hyperplane).
      a^T (p + dt*B(theta) u) >= b + margin   <=>   -a^T B u <= (a^T p - b - margin)/dt
    """
    B = np.array([[np.cos(theta), 0.0], [np.sin(theta), 0.0]])
    rows, rhs = [], []
    for A, b in OBS:
        s = A @ p - b                    # >0 on some row  <=> outside the obstacle
        i = int(np.argmax(s))
        rows.append(-A[i] @ B)
        rhs.append((s[i] - MARGIN) / DT)
    return np.array(rows), np.array(rhs)


def simulate(mode, seed=0, x0=np.array([-4.5, 0.0, 0.5])):
    rng = np.random.default_rng(seed)
    # reference: straight run to the goal at (4, 0) (unspecified in the paper; fixed
    # identically for all three controllers so the comparison is controlled)
    x = x0 + (rng.normal(size=3) * np.array([0.05, 0.05, 0.02]) if seed else 0.0)
    integ = np.zeros(3); prev = np.zeros(3)
    collisions, viol_max, cost = 0, 0.0, 0.0
    traj = []
    for t in range(STEPS):
        xr = np.array([min(-4.5 + 0.4 * DT * t * 10, 4.0), 0.0, 0.0])
        th = x[2]
        R = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
        e = R @ (xr - x)
        integ += e * DT
        de = (e - prev) / DT
        prev = e
        u = SEL @ (KP @ e + KI @ integ + KD @ de)
        As, bs = safety_rows(x[:2], th)
        A = np.vstack([AU, As]); b = np.concatenate([BU, bs])   # m = 7 > n_out = 2
        if mode == "soft":
            y = np.clip(u, [-0.01, -0.5], [1.0, 0.5])           # box only, no safety proj
        elif mode == "hardnet_like":
            y = hardnet_like(u, A, b)
        elif mode == "caffnet":
            y = caffnet(u, A, b, np.zeros(2), p=2)
        elif mode == "caffnet_k1_mutation":
            y = caffnet(u, A, b, np.zeros(2), p=2, kmax=1)
        else:
            raise ValueError(mode)
        viol_max = max(viol_max, float(np.max(A @ y - b)))
        cost += float(e @ e + y @ y) * DT
        x = x + DT * np.array([y[0] * np.cos(th), y[0] * np.sin(th), y[1]])
        traj.append(x[:2].tolist())
        if any(inside(x[:2], Ao, bo) for Ao, bo in OBS):
            collisions += 1
    return dict(mode=mode, seed=seed, collision_steps=collisions,
                collided=bool(collisions > 0), max_constraint_violation=viol_max,
                cost=cost, final_x=x.tolist())


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    for mode in ("soft", "hardnet_like", "caffnet", "caffnet_k1_mutation"):
        runs = [simulate(mode, s) for s in range(5)]
        out[mode] = dict(runs=runs,
                         n_seeds_collided=int(sum(r["collided"] for r in runs)),
                         total_collision_steps=int(sum(r["collision_steps"] for r in runs)),
                         max_constraint_violation=float(max(r["max_constraint_violation"] for r in runs)),
                         mean_cost=float(np.mean([r["cost"] for r in runs])))
        print(mode, out[mode]["n_seeds_collided"], out[mode]["max_constraint_violation"], flush=True)
    res = dict(claim=5, note="mechanism-level test; the paper's trained policies of Table 4 "
                             "are NOT reproduced (missing spec, see logbook)",
               dt=DT, steps=STEPS, margin=MARGIN, results=out,
               command="python verify_claim5.py", seconds=round(time.time() - t0, 2))
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
