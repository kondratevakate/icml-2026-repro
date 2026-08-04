"""Claim 5 (Section 4.3 / Appendix D.3): in the safety-critical control experiment CAffNet
avoids the obstacles, whereas HardNet and the soft-constrained baseline fail to do so.

Setup reproduced from Appendix D.3 (see notes_paper.md):
 unicycle xdot = g(theta) u, u = [v, omega]; dt = 0.1 s, horizon 15 s (150 steps);
 goal (0,0), arrival radius 0.1 m; state box -5<=px<=1, -4<=py<=2, -pi<=theta<=pi;
 control box -0.01<=v<=1, -0.5<=omega<=0.5 (A_u, b_u as in D.3);
 3 polytopic obstacles (A_j, b_j from D.3), per-edge CBF h_j^i = a_j^i p - b_j^i,
 smooth union h_j = (1/k) ln sum exp(k h_j^i) - ln(m_j)/k with k = 10;
 CBF condition Lf h + Lg h u >= -alpha(h), alpha(h) = h;
 aggregated affine constraint A(x) u <= b(x) with m = 3 obstacles + 6 state rows + 4 control
 rows = 13 and n_out = 2 (so min(m, n_out) = 2, |Gamma| = 13 + 78 = 91);
 u = u_nom(PID) + u_net; cost J with Q = diag(1000,1000,0), R = I, QN = diag(1e6,1e6,0).

Deviation from the paper (documented in logbook.md): fewer training initial states and
fewer epochs than the paper's 300 states, and HardNet is NOT reproduced because its
projection requires A(x) to have full row rank, which fails here (13 x 2).

Run: ./.venv/bin/python verify_claim5.py [--epochs N]
"""
import argparse, itertools, json, os
import numpy as np
import torch

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claim5.json")
DT, KAPPA = 0.1, 10.0

A1 = torch.tensor([[0.4472, -0.8944], [0.7071, 0.7071], [-0.2425, 0.9701],
                   [-0.7071, -0.7071], [-0.8944, -0.4472]], dtype=torch.float64)
b1 = torch.tensor([-0.2184, -0.5303, 0.6219, 1.1667, 1.4368], dtype=torch.float64)
A2 = torch.tensor([[-0.9685, 0.2489], [0.9417, 0.3363], [-0.3714, 0.9285],
                   [0.3714, 0.9285], [-0.9417, -0.3363], [-0.2976, -0.9547]], dtype=torch.float64)
b2 = torch.tensor([1.2755, -1.7670, -0.8511, -2.0249, 2.3274, 2.6868], dtype=torch.float64)
A3 = torch.tensor([[-0.9191, 0.3939], [0.8944, 0.4472], [0.9703, -0.2419],
                   [-0.8701, -0.4930], [0.0000, -1.0000]], dtype=torch.float64)
b3 = torch.tensor([2.9916, -1.9975, -2.5305, 2.4854, 0.1000], dtype=torch.float64)
OBS = [(A1, b1), (A2, b2), (A3, b3)]
Au = torch.tensor([[1.0, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float64)
bu = torch.tensor([1.0, 0.01, 0.5, 0.5], dtype=torch.float64)
Ax = torch.tensor([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                  dtype=torch.float64)
bx = torch.tensor([1.0, 5.0, 2.0, 4.0, np.pi, np.pi], dtype=torch.float64)

GAMMAS = [g for k in (1, 2) for g in itertools.combinations(range(13), k)]


def gmat(th):
    c, s, z, o = torch.cos(th), torch.sin(th), torch.zeros_like(th), torch.ones_like(th)
    return torch.stack([torch.stack([c, z], -1), torch.stack([s, z], -1),
                        torch.stack([z, o], -1)], -2)          # B x 3 x 2


def inside_obstacle(p):
    """True if position p (B x 2) is inside any obstacle (all edges satisfied)."""
    out = torch.zeros(p.shape[0], dtype=torch.bool)
    for Aj, bj in OBS:
        out |= ((p @ Aj.T - bj) <= 0).all(dim=1)
    return out


def constraints(x):
    """A(x) in R^{B x 13 x 2}, b(x) in R^{B x 13}."""
    p, th = x[:, :2], x[:, 2]
    G = gmat(th)
    rowsA, rowsb = [], []
    for Aj, bj in OBS:
        hi = p @ Aj.T - bj                                     # B x m_j
        hj = torch.logsumexp(KAPPA * hi, dim=1) / KAPPA - np.log(Aj.shape[0]) / KAPPA
        lam = torch.exp(KAPPA * (hi - hj.unsqueeze(1)))        # B x m_j
        grad_p = lam @ Aj                                      # B x 2 (d h_j / d p)
        grad = torch.cat([grad_p, torch.zeros(p.shape[0], 1, dtype=p.dtype)], dim=1)  # B x 3
        Lg = torch.bmm(grad.unsqueeze(1), G).squeeze(1)        # B x 2, Lf = 0
        rowsA.append(-Lg); rowsb.append(hj)                    # alpha(h) = h
    hx = bx - x @ Ax.T                                         # B x 6
    for i in range(6):
        grad = -Ax[i].expand(x.shape[0], 3)
        Lg = torch.bmm(grad.unsqueeze(1), G).squeeze(1)
        rowsA.append(-Lg); rowsb.append(hx[:, i])
    A = torch.stack(rowsA, dim=1)                              # B x 9 x 2
    b = torch.stack(rowsb, dim=1)
    A = torch.cat([A, Au.expand(x.shape[0], 4, 2)], dim=1)     # B x 13 x 2
    b = torch.cat([b, bu.expand(x.shape[0], 4)], dim=1)
    return A, b


def caffine(f, A, b, w):
    """Eq (12), batched. f,w: B x 2; A: B x 13 x 2; b: B x 13."""
    B = f.shape[0]
    feas_f = ((torch.bmm(A, f.unsqueeze(2)).squeeze(2) - b) <= 1e-9).all(dim=1, keepdim=True)
    cands, ok = [], []
    for g in GAMMAS:
        idx = list(g)
        Ag, bg = A[:, idx, :], b[:, idx]
        Agp = torch.linalg.pinv(Ag)
        y = f - torch.bmm(Agp, (torch.bmm(Ag, f.unsqueeze(2)).squeeze(2) - bg).unsqueeze(2)).squeeze(2)
        N = torch.eye(2, dtype=f.dtype).expand(B, 2, 2) - torch.bmm(Agp, Ag)
        y = y + torch.bmm(N, w.unsqueeze(2)).squeeze(2)
        cands.append(y)
        ok.append(((torch.bmm(A, y.unsqueeze(2)).squeeze(2) - b) <= 1e-7).all(dim=1))
    C = torch.stack(cands, 1)                                   # B x G x 2
    OKm = torch.stack(ok, 1)                                    # B x G
    d = (C - f.unsqueeze(1)).norm(dim=2).masked_fill(~OKm, float("inf"))
    idx = d.argmin(dim=1)
    proj = C[torch.arange(B), idx]
    none = ~OKm.any(dim=1)
    y = torch.where(feas_f, f, proj)
    return y, none


def u_nom(x):
    p, th = x[:, :2], x[:, 2]
    d = p.norm(dim=1)
    desired = torch.atan2(-p[:, 1], -p[:, 0])
    err = torch.atan2(torch.sin(desired - th), torch.cos(desired - th))
    v = torch.clamp(0.8 * d, -0.01, 1.0)
    w = torch.clamp(1.5 * err, -0.5, 0.5)
    return torch.stack([v, w], dim=1)


class Net(torch.nn.Module):
    def __init__(self, out=2):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(3, 200), torch.nn.ReLU(),
                                     torch.nn.Linear(200, 200), torch.nn.ReLU(),
                                     torch.nn.Linear(200, 200), torch.nn.ReLU(),
                                     torch.nn.Linear(200, out))
        self.w = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.ReLU(),
                                     torch.nn.Linear(64, out))

    def forward(self, x):
        return self.f(x), self.w(x)


def rollout(net, x0, mode, steps=150, grad=True, apost=False):
    """mode: 'soft' | 'caff'. Returns cost, collisions, control-violations, arrivals."""
    x = x0.clone()
    Q = torch.tensor([1000.0, 1000.0, 0.0], dtype=torch.float64)
    cost = torch.zeros((), dtype=torch.float64)
    collided = torch.zeros(x.shape[0], dtype=torch.bool)
    uviol = torch.zeros(x.shape[0], dtype=torch.bool)
    infeas = 0
    for k in range(steps):
        A, b = constraints(x)
        f, w = net(x)
        un = u_nom(x)
        if mode == "caff" and not apost:
            u, none = caffine(un + f, A, b, w)
            infeas += int(none.sum())
        elif mode == "caff" and apost:
            u = un + f
            u, none = caffine(u.detach(), A, b, w.detach())
        else:
            u = un + f
        pen = 100 * (torch.bmm(A, u.unsqueeze(2)).squeeze(2) - b).clamp(min=0)
        cost = cost + (x ** 2 * Q).sum() + (f ** 2).sum() + (pen.sum() if mode == "soft" else 0.0)
        with torch.no_grad():
            uv = (u @ Au.T - bu) > 1e-7
            uviol |= uv.any(dim=1)
        x = x + DT * torch.bmm(gmat(x[:, 2]), u.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            collided |= inside_obstacle(x[:, :2])
    cost = cost + 1e6 * (x[:, :2] ** 2).sum()
    arrived = (x[:, :2].norm(dim=1) <= 0.5)
    return cost, collided, uviol, arrived, infeas, x


def sample_states(n, seed):
    rng = np.random.default_rng(seed)
    px = rng.uniform(-5, 1, n); py = rng.uniform(-4, 2, n)
    keep = []
    for i in range(n):
        p = torch.tensor([[px[i], py[i]]], dtype=torch.float64)
        if not inside_obstacle(p)[0] and np.hypot(px[i], py[i]) > 1.0:
            keep.append((px[i], py[i]))
    px = np.array([k[0] for k in keep]); py = np.array([k[1] for k in keep])
    th = np.arctan2(-py, -px)
    return torch.tensor(np.stack([px, py, th], 1), dtype=torch.float64)


def train(mode, seed, epochs, ntrain, apost=False):
    torch.manual_seed(seed)
    x0 = sample_states(ntrain, 1000 + seed)
    net = Net().double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    for e in range(epochs):
        opt.zero_grad()
        cost, *_ = rollout(net, x0, mode, steps=150, apost=apost)
        (cost / x0.shape[0]).backward()
        opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ntrain", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    xeval = sample_states(60, 999)
    runs = []
    for mode, apost in [("soft", False), ("caff", False), ("caff", True)]:
        name = {"soft": "NN-soft", "caff": "CAffNet-FF"}[mode] + ("-aposteriori" if apost else "")
        for seed in range(a.seeds):
            net = train(mode, seed, a.epochs, a.ntrain, apost=apost)
            with torch.no_grad():
                cost, coll, uviol, arrived, infeas, xf = rollout(net, xeval, mode, grad=False,
                                                                 apost=apost)
            r = {"method": name, "seed": seed,
                 "n_eval_states": int(xeval.shape[0]),
                 "collisions": int(coll.sum()), "collision_rate": float(coll.float().mean()),
                 "control_violations": int(uviol.sum()),
                 "arrived_within_0.5m": int(arrived.sum()),
                 "infeasible_steps": infeas,
                 "final_cost": float(cost)}
            runs.append(r); print(r, flush=True)
    agg = {}
    for name in sorted({r["method"] for r in runs}):
        rs = [r for r in runs if r["method"] == name]
        agg[name] = {"mean_collisions": float(np.mean([r["collisions"] for r in rs])),
                     "mean_control_violations": float(np.mean([r["control_violations"] for r in rs])),
                     "mean_arrived": float(np.mean([r["arrived_within_0.5m"] for r in rs]))}
    res = {"claim": 5,
           "source": "Section 4.3 + Appendix D.3 (Fig 6/7, Table 4)",
           "command": f"./.venv/bin/python verify_claim5.py --epochs {a.epochs} --ntrain {a.ntrain} --seeds {a.seeds}",
           "epochs": a.epochs, "ntrain": a.ntrain, "seeds": a.seeds,
           "n_gammas": len(GAMMAS), "m_constraints": 13, "n_out": 2,
           "hardnet_not_reproduced_reason":
               "HardNet's projection needs A(x) with full row rank; here A(x) is 13x2 "
               "(rank 2 << 13), so the HardNet formula is undefined on this problem.",
           "runs": runs, "aggregate": agg}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
