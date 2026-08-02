"""verify_claim5.py — Section 4.3 / Table 4: safety-critical control.

Claim: CAffNet avoids the obstacles whereas both HardNet and the soft-constrained
baseline fail to do so.

Implementation follows Appendix D.3 exactly where the paper specifies it
(see notes_paper.md): unicycle dynamics, PID nominal controller with the paper's
gains, saturation, the three polytopic obstacles A_j/b_j, the smooth-union CBF
h_j(x) = (1/kappa) ln sum_i exp(kappa h_j^i(x)) - ln(m_j)/kappa with kappa=10,
alpha(h)=h, state-constraint CBFs, control bounds A_u u <= b_u, dt = 0.1 s,
15 s rollouts, cost J with Q=diag(1000,1000,0), R=I_2, Q_N=diag(1e6,1e6,0),
test initial state x0 = [-4.5, 0, 0.5].

Aggregated constraint (Eq. in Sec. 4.3):
  A(x) = [-Lg h_1; -Lg h_2; -Lg h_3; -Lg h_x; A_u]   (m = 3 + 6 + 4 = 13, n_out = 2)
  b(x) = [Lf h_1 + h_1; ...; Lf h_x + h_x; b_u]
Note f(x) = 0 for the unicycle, so Lf h == 0.

Three policies are trained with identical budget/seeds:
  NN       : soft penalty 100 * ReLU(A u - b)
  HardNet  : pseudo-inverse correction u = u_net - A^dag ReLU(A u_net - b) + soft penalty
             (A(x) here has m=13 > n_out=2 rows, i.e. NOT full row rank -> HardNet's
              assumption is violated, which is the paper's point)
  CAffNet  : CAffine layer, Gamma = all k in {1,2} subsets of the 13 rows (91 combos),
             with the trainable null-space component w_phi(x)

SCALE: the paper trains on 300 initial states with a GPU. This script exposes
--n-init / --epochs; whatever is used is recorded in results/claim5.json and the
verdict in logbook.md is qualified accordingly. No number is reported as matching
Table 4 unless it actually does.

Run: .venv/bin/python verify_claim5.py --epochs 60 --n-init 40
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

A1 = torch.tensor([[0.4472, -0.8944], [0.7071, 0.7071], [-0.2425, 0.9701],
                   [-0.7071, -0.7071], [-0.8944, -0.4472]])
b1 = torch.tensor([-0.2184, -0.5303, 0.6219, 1.1667, 1.4368])
A2 = torch.tensor([[-0.9685, 0.2489], [0.9417, 0.3363], [-0.3714, 0.9285],
                   [0.3714, 0.9285], [-0.9417, -0.3363], [-0.2976, -0.9547]])
b2 = torch.tensor([1.2755, -1.7670, -0.8511, -2.0249, 2.3274, 2.6868])
A3 = torch.tensor([[-0.9191, 0.3939], [0.8944, 0.4472], [0.9703, -0.2419],
                   [-0.8701, -0.4930], [0.0000, -1.0000]])
b3 = torch.tensor([2.9916, -1.9975, -2.5305, 2.4854, 0.1000])
OBS = [(A1, b1), (A2, b2), (A3, b3)]
KAPPA = 10.0

Ax = torch.tensor([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
bx = torch.tensor([1., 5., 2., 4., np.pi, np.pi])
Au = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
bu = torch.tensor([1.0, 0.01, 0.5, 0.5])

Kp = torch.tensor([0.01, 0.2, 0.0])
Ki = torch.tensor([0.05, 0.005, 0.0])
Kd = torch.tensor([0.0, 0.01, 0.0])
SEL = torch.tensor([[1., 0., 0.], [0., 1., 1.]])
DT = 0.1


def g_mat(theta):
    """g(x) in R^{3x2}, batched: (B,3,2)."""
    B = theta.shape[0]
    G = torch.zeros(B, 3, 2, dtype=theta.dtype)
    G[:, 0, 0] = torch.cos(theta)
    G[:, 1, 0] = torch.sin(theta)
    G[:, 2, 1] = 1.0
    return G


def obstacle_cbf(x):
    """Returns h (B,3) and Lg h (B,3,2) for the three smooth-union obstacle CBFs."""
    p = x[:, :2]
    G = g_mat(x[:, 2])
    hs, Lgs = [], []
    for Aj, bj in OBS:
        hij = p @ Aj.T - bj                      # (B,mj) ; h_j^i = a_j^i x - b_j^i
        mj = Aj.shape[0]
        hj = (torch.logsumexp(KAPPA * hij, dim=1) - np.log(mj)) / KAPPA   # (B,)
        lam = torch.exp(KAPPA * (hij - hj.unsqueeze(1)))                  # (B,mj)
        # grad_x h_j^i = [a_j^i, 0] in R^3 ; Lg h^i = grad^T g
        gradi = torch.zeros(x.shape[0], mj, 3, dtype=x.dtype)
        gradi[:, :, :2] = Aj.unsqueeze(0).expand(x.shape[0], -1, -1)
        grad = (lam.unsqueeze(2) * gradi).sum(dim=1)                      # (B,3)
        Lg = torch.bmm(grad.unsqueeze(1), G).squeeze(1)                   # (B,2)
        hs.append(hj)
        Lgs.append(Lg)
    return torch.stack(hs, 1), torch.stack(Lgs, 1)


def state_cbf(x):
    """h_x = b_x - A_x x >= 0, six rows; Lg h_x = -A_x g."""
    h = bx.unsqueeze(0) - x @ Ax.T                     # (B,6)
    G = g_mat(x[:, 2])
    Lg = -torch.einsum('ij,bjk->bik', Ax, G)           # (B,6,2)
    return h, Lg


def constraints(x):
    """Aggregated A(x) u <= b(x); returns A (B,13,2), b (B,13). f(x)=0 => Lf h = 0."""
    ho, Lgo = obstacle_cbf(x)
    hx, Lgx = state_cbf(x)
    B = x.shape[0]
    A = torch.cat([-Lgo, -Lgx, Au.unsqueeze(0).expand(B, -1, -1)], dim=1)
    b = torch.cat([ho, hx, bu.unsqueeze(0).expand(B, -1)], dim=1)
    return A, b


# ---- CAffine layer, general (m=13, n_out=2 => Gamma = C(13,1)+C(13,2) = 91) ----
import itertools
GAMMA = [c for k in (1, 2) for c in itertools.combinations(range(13), k)]


def caffine(u, w, A, b, tol=1e-9):
    """Eq. (8),(9),(12) batched. u,w: (B,2); A: (B,13,2); b: (B,13)."""
    B = u.shape[0]
    feas0 = ((torch.bmm(A, u.unsqueeze(2)).squeeze(2) - b) <= tol).all(dim=1)
    cands = []
    for g in GAMMA:
        idx = list(g)
        Ag = A[:, idx, :]                              # (B,k,2)
        bg = b[:, idx]
        Agd = torch.linalg.pinv(Ag)                    # (B,2,k)
        corr = torch.bmm(Agd, (torch.bmm(Ag, u.unsqueeze(2)).squeeze(2) - bg).unsqueeze(2)).squeeze(2)
        Nn = torch.eye(2, dtype=u.dtype).unsqueeze(0) - torch.bmm(Agd, Ag)
        cands.append(u - corr + torch.bmm(Nn, w.unsqueeze(2)).squeeze(2))
    C = torch.stack(cands, 1)                          # (B,|G|,2)
    viol = torch.relu(torch.einsum('bmn,bgn->bgm', A, C) - b.unsqueeze(1))
    ok = (viol <= 1e-7).all(dim=2)                     # (B,|G|)
    d = torch.linalg.norm(C - u.unsqueeze(1), dim=2)
    d = torch.where(ok, d, torch.full_like(d, 1e9))
    i = d.argmin(dim=1)
    proj = C[torch.arange(B), i]
    any_ok = ok.any(dim=1)
    out = torch.where(feas0.unsqueeze(1), u, proj)
    return out, any_ok


def hardnet(u, A, b):
    v = torch.relu(torch.bmm(A, u.unsqueeze(2)).squeeze(2) - b)
    Ad = torch.linalg.pinv(A)
    return u - torch.bmm(Ad, v.unsqueeze(2)).squeeze(2)


class Net(nn.Module):
    def __init__(self, nout=2, h=200):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, h), nn.ReLU(), nn.Linear(h, nout))

    def forward(self, x):
        return self.f(x)


def pid(x, integ, prev_e):
    """u_nom from the PID controller (reference = origin, x_ref = 0)."""
    th = x[:, 2]
    R = torch.zeros(x.shape[0], 3, 3, dtype=x.dtype)
    R[:, 0, 0] = torch.cos(th); R[:, 0, 1] = torch.sin(th)
    R[:, 1, 0] = -torch.sin(th); R[:, 1, 1] = torch.cos(th)
    R[:, 2, 2] = 1.0
    e = torch.bmm(R, (-x).unsqueeze(2)).squeeze(2)          # x_ref = 0
    integ = integ + e * DT
    de = (e - prev_e) / DT
    term = Kp * e + Ki * integ + Kd * de
    u = term @ SEL.T
    return u, integ, e


def sat_u(u):
    v = torch.clamp(u[:, 0:1], -0.01, 1.0)
    w = torch.clamp(u[:, 1:2], -0.5, 0.5)
    return torch.cat([v, w], dim=1)


def rollout(net, wnet, method, x0, steps, grad=True):
    x = x0.clone()
    integ = torch.zeros_like(x)
    prev_e = torch.zeros_like(x)
    cost = torch.zeros((), dtype=x.dtype)
    Q = torch.tensor([1000., 1000., 0.])
    viol_max = 0.0
    viol_sum = 0.0
    nviol = 0
    ncheck = 0
    n_infeasible = 0
    traj = []
    for k in range(steps):
        un, integ, prev_e = pid(x, integ, prev_e)
        un = sat_u(un)
        A, b = constraints(x)
        raw = net(x)
        # the constraints act on the TOTAL command u = u_nom + u_net (Sec. 4.3),
        # so the hard layers are applied to u_nom + raw and u_net is the correction
        if method == "CAffNet":
            u, ok = caffine(un + raw, wnet(x), A, b)
            n_infeasible += int((~ok).sum())
            u_net = u - un
        elif method == "HardNet":
            u = hardnet(un + raw, A, b)
            u_net = u - un
        else:
            u_net = raw
            u = sat_u(un + u_net)
        r = torch.relu(torch.bmm(A, u.unsqueeze(2)).squeeze(2) - b)
        viol_max = max(viol_max, float(r.max()))
        viol_sum += float(r.mean())
        nviol += int((r > 1e-7).any(dim=1).sum())
        ncheck += x.shape[0]
        cost = cost + (x ** 2 * Q).sum() + (u_net ** 2).sum()
        if method in ("NN", "HardNet"):
            cost = cost + 100 * r.mean() * x.shape[0]
        G = g_mat(x[:, 2])
        x = x + DT * torch.bmm(G, u.unsqueeze(2)).squeeze(2)
        traj.append(x.detach().clone())
    cost = cost + (x ** 2 * torch.tensor([1e6, 1e6, 0.])).sum()
    return cost, dict(viol_max=viol_max, viol_mean=viol_sum / steps,
                      viol_pct=100.0 * nviol / max(ncheck, 1),
                      n_infeasible_states=n_infeasible), traj


def inside_obstacle(p):
    """True if point p (2,) is strictly inside any obstacle polytope."""
    for Aj, bj in OBS:
        if bool((Aj @ torch.tensor(p, dtype=torch.float32) <= bj).all()):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-init", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sim-s", type=float, default=15.0)
    a = ap.parse_args()
    steps = int(a.sim_s / DT)
    t0 = time.time()
    recs = []
    for method in ["NN", "HardNet", "CAffNet"]:
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            px = rng.uniform(-5, 1, a.n_init)
            py = rng.uniform(-4, 2, a.n_init)
            th = np.arctan2(-py, -px)                    # oriented toward origin
            X0 = torch.tensor(np.stack([px, py, th], 1), dtype=torch.float32)
            net, wnet = Net(), Net()
            params = list(net.parameters()) + (list(wnet.parameters())
                                               if method == "CAffNet" else [])
            opt = torch.optim.Adam(params, lr=1e-4)
            for ep in range(a.epochs):
                opt.zero_grad()
                cost, _, _ = rollout(net, wnet, method, X0, steps)
                (cost / a.n_init).backward()
                opt.step()
                if ep % 10 == 0:
                    print(f"{method} s{seed} ep{ep} cost/init={float(cost)/a.n_init:.1f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            with torch.no_grad():
                x0 = torch.tensor([[-4.5, 0.0, 0.5]])
                c, m, traj = rollout(net, wnet, method, x0, steps)
                pts = [t[0, :2].numpy().tolist() for t in traj]
                coll = any(inside_obstacle(p) for p in pts)
                dmin = min(float(np.hypot(*p)) for p in pts)
                ctr, mtr, _ = rollout(net, wnet, method, X0, steps)
            recs.append(dict(method=method, seed=seed,
                             test_cost=float(c), collision=bool(coll),
                             min_dist_to_goal=dmin, reached_goal=bool(dmin <= 0.1),
                             test_viol_max=m["viol_max"], test_viol_mean=m["viol_mean"],
                             test_viol_pct=m["viol_pct"],
                             train_set_cost_per_init=float(ctr) / a.n_init,
                             train_set_viol_max=mtr["viol_max"],
                             train_set_viol_mean=mtr["viol_mean"],
                             train_set_viol_pct=mtr["viol_pct"],
                             traj_xy=pts))
            print(f"DONE {method} s{seed}: collision={coll} dmin={dmin:.3f} "
                  f"vmax={m['viol_max']:.4f}", flush=True)
    agg = {}
    for meth in ["NN", "HardNet", "CAffNet"]:
        rs = [r for r in recs if r["method"] == meth]
        agg[meth] = dict(
            n_seeds=len(rs),
            n_collisions=sum(r["collision"] for r in rs),
            n_reached_goal=sum(r["reached_goal"] for r in rs),
            mean_test_viol_max=float(np.mean([r["test_viol_max"] for r in rs])),
            mean_train_viol_pct=float(np.mean([r["train_set_viol_pct"] for r in rs])),
            mean_cost_per_init=float(np.mean([r["train_set_cost_per_init"] for r in rs])))
    out = {"claim": 5,
           "claim_text": "CAffNet avoids obstacles while HardNet and the soft baseline fail",
           "source": "Section 4.3, Table 4, Fig. 6; setup Appendix D.3",
           "command": (f".venv/bin/python verify_claim5.py --epochs {a.epochs} "
                       f"--n-init {a.n_init} --seeds {a.seeds}"),
           "scale_note": ("PAPER: 300 initial states, GPU, unspecified epoch count. "
                          f"THIS RUN: {a.n_init} initial states, {a.epochs} epochs, "
                          f"{a.seeds} seeds, CPU. Reduced scale — see logbook."),
           "paper_table4": {"NN_cost": 4.1411e5, "HardNet_cost": 4.5701e5,
                            "CAffNet_cost": 7.2060e5, "NN_viol_pct": 2.60,
                            "HardNet_viol_pct": 2.60, "CAffNet_viol_pct": 0.0},
           "aggregate": agg, "records": recs, "elapsed_s": time.time() - t0}
    json.dump(out, open("results/claim5.json", "w"), indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
