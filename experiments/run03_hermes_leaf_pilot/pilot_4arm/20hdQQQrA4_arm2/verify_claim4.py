"""verify_claim4.py — piecewise-constraint benchmark: 73.33% MSE reduction, zero violations.

CLAIM (Experiments, piecewise constraints task): CAffNet-TF achieves a 73.33% MSE
reduction relative to a soft-constrained neural-network baseline while incurring zero
constraint violations.

ROUTE (skill §2g): the claim is TWO sub-claims with different reproducibility.
  * STRUCTURAL half — zero constraint violations. Produced by the architecture, scale
    free, genuinely verifiable on CPU at reduced scale.
  * MAGNITUDE half — the 73.33% figure. Depends on the paper's transformer variant
    trained at GPU scale on the paper's own (unreleased) piecewise benchmark. NOT
    reproducible in this budget; reported as `toy` with the scale deficit stated.
  * Paper-internal arithmetic: 73.33% is recomputed as 1 - 4/15 = 1 - 0.0012/0.0045,
    i.e. the quoted percentage is at least internally consistent with a 4:15 MSE ratio.

SETUP (independent reimplementation, numpy only): x ~ U[-1,1]; target f_t(x) in R^2 is a
piecewise-linear function that is feasible for a piecewise-defined constraint system
A(x) y <= b(x) (two half-planes whose coefficients switch sign at x = 0, plus a box).
Both models are the same 1-32-32-2 tanh MLP trained with Adam; they differ ONLY in the
output stage:
  * soft  : raw output, loss = MSE + mu * mean(relu(A y - b)^2)
  * caff  : output passed through the CAffine layer (w = 0). The layer is affine in f on
            each selection branch (y = (I - A_G^+ A_G) f + A_G^+ b_G), so its Jacobian is
            exactly that matrix and back-propagation is exact a.e.

MUTATION: remove the CAffine layer and keep the identical training (the `soft` arm).
PREDICTION (before running): the soft arm shows a non-zero violation rate on the test
grid, the CAffine arm shows exactly zero.

SCALE NOTE: PAPER: transformer variant (CAffNet-TF), full benchmark, GPU.
THIS RUN: 1-32-32-2 MLP, 1500 Adam steps, 3 seeds, 8 vCPU, no GPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from caff_core import feasible, max_violation, pinv
from vec_caff import VecCaff

CMD = "python verify_claim4.py"
SEEDS = [0, 1, 2]
STEPS = 1500
MU = 1.0
N_TRAIN, N_TEST = 512, 400


def constraints(x):
    """Piecewise-defined constraint system A(x) y <= b(x), y in R^2 (m = 4)."""
    s = 1.0 if x >= 0 else -1.0
    A = np.array([[s, 1.0],
                  [-s, 0.5],
                  [1.0, 0.0],
                  [0.0, -1.0]])
    b = np.array([0.8 + 0.2 * x, 0.9, 1.2, 1.2])
    return A, b


def target(x):
    """Piecewise-linear target lying EXACTLY on the first (piecewise) constraint.

    The paper's benchmark is a constrained-regression task whose optimum is active, so
    the target is placed on the boundary: then any residual approximation error of an
    unconstrained/soft model shows up as a constraint violation, which is precisely what
    the structural half of the claim is about.
    """
    A, b = constraints(x)
    y = np.array([0.3 * x, -0.4 * abs(x) + 0.1])
    a0, b0 = A[0], b[0]
    y = y + (b0 - a0 @ y) * a0 / (a0 @ a0)      # push onto row 0 boundary
    if not feasible(A, b, y):                    # keep it inside the other rows
        y = np.clip(y, -1.0, 1.0)
        y = y + (b0 - a0 @ y) * a0 / (a0 @ a0)
    return y


_A_POS, _ = constraints(1.0)
_A_NEG, _ = constraints(-1.0)
_LAYERS = {1.0: VecCaff(_A_POS), -1.0: VecCaff(_A_NEG)}


def caff_batch(X, F):
    """Apply the CAffine layer to a whole batch. Returns (Y, per-sample Jacobians)."""
    Y = np.empty_like(F)
    J = np.repeat(np.eye(F.shape[1])[None], F.shape[0], axis=0)
    for s in (1.0, -1.0):
        mask = (X[:, 0] >= 0) if s > 0 else (X[:, 0] < 0)
        if not np.any(mask):
            continue
        B = np.stack([constraints(float(x))[1] for x in X[mask, 0]])
        y, j = _LAYERS[s].apply(F[mask], B)
        Y[mask], J[mask] = y, j
    return Y, J


class MLP:
    def __init__(self, rng, h=32):
        self.W = [rng.normal(size=(1, h)) * 0.8, rng.normal(size=(h, h)) * (1 / np.sqrt(h)),
                  rng.normal(size=(h, 2)) * (1 / np.sqrt(h))]
        self.B = [np.zeros(h), np.zeros(h), np.zeros(2)]
        self.m = [np.zeros_like(p) for p in self.W + self.B]
        self.v = [np.zeros_like(p) for p in self.W + self.B]
        self.t = 0

    def forward(self, X):
        h1 = np.tanh(X @ self.W[0] + self.B[0])
        h2 = np.tanh(h1 @ self.W[1] + self.B[1])
        out = h2 @ self.W[2] + self.B[2]
        return out, (X, h1, h2)

    def backward(self, cache, dout):
        X, h1, h2 = cache
        gW2 = h2.T @ dout
        gB2 = dout.sum(0)
        d2 = (dout @ self.W[2].T) * (1 - h2 ** 2)
        gW1 = h1.T @ d2
        gB1 = d2.sum(0)
        d1 = (d2 @ self.W[1].T) * (1 - h1 ** 2)
        gW0 = X.T @ d1
        gB0 = d1.sum(0)
        return [gW0, gW1, gW2, gB0, gB1, gB2]

    def step(self, grads, lr=3e-3):
        self.t += 1
        params = self.W + self.B
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g ** 2
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)


def run_seed(seed):
    rng = np.random.default_rng([seed, 4])
    Xtr = rng.uniform(-1, 1, size=(N_TRAIN, 1))
    Ytr = np.stack([target(float(x)) for x in Xtr[:, 0]])
    Xte = np.linspace(-1, 1, N_TEST).reshape(-1, 1)
    Yte = np.stack([target(float(x)) for x in Xte[:, 0]])
    cons_tr = [constraints(float(x)) for x in Xtr[:, 0]]
    cons_te = [constraints(float(x)) for x in Xte[:, 0]]

    out = {}
    for arm in ("soft", "caff"):
        net = MLP(np.random.default_rng([seed, 4, 0 if arm == "soft" else 1]))
        for _ in range(STEPS):
            f, cache = net.forward(Xtr)
            if arm == "soft":
                d = 2.0 * (f - Ytr) / N_TRAIN
                for i, (A, b) in enumerate(cons_tr):
                    v = np.maximum(A @ f[i] - b, 0.0)
                    d[i] += MU * 2.0 * (A.T @ v) / N_TRAIN
            else:
                ys, Js = caff_batch(Xtr, f)
                dy = 2.0 * (ys - Ytr) / N_TRAIN
                d = np.einsum('nij,ni->nj', Js, dy)
            net.step(net.backward(cache, d))

        f, _ = net.forward(Xte)
        pred = caff_batch(Xte, f)[0] if arm == "caff" else f
        mse = float(np.mean((pred - Yte) ** 2))
        viol = [max_violation(A, b, pred[i]) for i, (A, b) in enumerate(cons_te)]
        out[arm] = {"mse": mse,
                    "n_violations": int(sum(v > 1e-7 for v in viol)),
                    "max_violation": float(max(viol)),
                    "n_test": N_TEST}
    return out


def main():
    t0 = time.time()
    per_seed = {}
    for s in SEEDS:
        per_seed[str(s)] = run_seed(s)
    mse_soft = float(np.mean([per_seed[str(s)]["soft"]["mse"] for s in SEEDS]))
    mse_caff = float(np.mean([per_seed[str(s)]["caff"]["mse"] for s in SEEDS]))
    red = 1.0 - mse_caff / mse_soft
    n_v_caff = int(sum(per_seed[str(s)]["caff"]["n_violations"] for s in SEEDS))
    n_v_soft = int(sum(per_seed[str(s)]["soft"]["n_violations"] for s in SEEDS))

    res = {
        "claim": 4,
        "source": "Experiments, piecewise-constraint benchmark (73.33% MSE reduction, zero violations)",
        "command": CMD,
        "scale_note": ("PAPER: CAffNet-TF transformer, full piecewise benchmark, GPU. "
                       "THIS RUN: 1-32-32-2 numpy MLP, 1500 Adam steps, 3 seeds, CPU-only."),
        "config": {"seeds": SEEDS, "steps": STEPS, "mu_soft_penalty": MU,
                   "n_train": N_TRAIN, "n_test": N_TEST},
        "structural": {
            "n_test_points_total": N_TEST * len(SEEDS),
            "caffnet_n_violations": n_v_caff,
            "caffnet_max_violation": float(max(per_seed[str(s)]["caff"]["max_violation"] for s in SEEDS)),
            "soft_baseline_n_violations": n_v_soft,
            "soft_baseline_max_violation": float(max(per_seed[str(s)]["soft"]["max_violation"] for s in SEEDS)),
            "verdict": None,
        },
        "magnitude": {
            "measured_mse_soft": mse_soft,
            "measured_mse_caffnet": mse_caff,
            "measured_reduction_pct": 100.0 * red,
            "paper_reduction_pct": 73.33,
            "paper_internal_arithmetic": {
                "check": "1 - 0.0012/0.0045",
                "value_pct": 100.0 * (1.0 - 0.0012 / 0.0045),
                "consistent_with_73_33": abs(100.0 * (1 - 0.0012 / 0.0045) - 73.33) < 0.01,
            },
            "verdict": "toy",
            "verdict_cap_reason": ("magnitude depends on the paper's transformer variant, its "
                                   "unreleased benchmark data and 50k-epoch GPU training; this "
                                   "run is a reduced-scale MLP surrogate"),
        },
        "mutation_remove_caffine_layer": {
            "prediction": "the soft-penalty arm shows a non-zero violation rate; CAffNet exactly zero",
            "soft_n_violations": n_v_soft,
            "caff_n_violations": n_v_caff,
        },
        "per_seed": per_seed,
        "verdict": None,
        "wall_seconds": None,
    }
    res["structural"]["verdict"] = "verified" if n_v_caff == 0 else "falsified"
    res["verdict"] = f"{res['structural']['verdict']} (structural) / toy (magnitude)"
    res["wall_seconds"] = round(time.time() - t0, 2)
    Path("results").mkdir(exist_ok=True)
    Path("results/claim4.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
