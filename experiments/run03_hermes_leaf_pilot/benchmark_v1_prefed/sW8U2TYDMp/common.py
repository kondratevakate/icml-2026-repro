"""Shared utilities for sW8U2TYDMp reproduction (CPU, numpy only)."""
import json, os
import numpy as np

SEED = 20260802
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def rng(offset=0):
    return np.random.default_rng(SEED + offset)


def norm(p):
    p = np.asarray(p, float)
    return p / p.sum()


def log_pool(Ps, beta):
    L = np.tensordot(np.asarray(beta, float), np.log(np.asarray(Ps, float)), axes=1)
    L -= L.max()
    p = np.exp(L)
    return p / p.sum()


def lin_pool(Ps, beta):
    return np.tensordot(np.asarray(beta, float), np.asarray(Ps, float), axes=1)


def H(p):
    return float(-(p * np.log(p)).sum())


def KL(p, q):
    return float((p * np.log(p / q)).sum())


def delta(Pi, P):
    """Welfare gap Delta_{Pi}(P) = E_P[log Pi] - E_{Pi}[log Pi] (Def. 7)."""
    return float((P * np.log(Pi)).sum() - (Pi * np.log(Pi)).sum())


def rand_simplex(r, k, conc=1.0):
    return r.dirichlet(np.full(k, conc))


def save(n, payload):
    os.makedirs(RESULTS, exist_ok=True)
    payload.setdefault("seed", SEED)
    with open(os.path.join(RESULTS, f"claim{n}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
