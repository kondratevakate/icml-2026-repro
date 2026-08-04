"""Shared stochastic-environment learning harness for the regret claims (1, 3, 5)."""
import numpy as np


def sample_instance(rng, T, K, values, sbound=2):
    arr = []
    for t in range(T):
        n = int(rng.integers(0, 3))
        step = []
        for _ in range(n):
            v = float(values[rng.integers(K)])
            d = min(t + int(rng.integers(0, sbound)), T - 1)
            step.append((v, d))
        arr.append(step)
    return arr


def run_alg_theta(arr, T, theta):
    pending, tot = [], 0.0
    for t in range(T):
        pending = [q for q in pending if q[1] >= t] + list(arr[t])
        if not pending:
            continue
        exp_now = [q for q in pending if q[1] == t]
        h = max(pending, key=lambda q: q[0])
        if exp_now:
            e = max(exp_now, key=lambda q: q[0])
            send = h if h[0] >= theta * e[0] else e
        else:
            send = h
        tot += send[0]
        pending.remove(send)
    return tot


def opt_dp(arr, T, sbound=2):
    """Exact offline optimum via DP over the (small) pending pool, kept tractable by
    dominance pruning: only the sbound-1 highest-value carried packets can matter."""
    states = {(): 0.0}
    keep_n = sbound - 1
    for t in range(T):
        nxt = {}
        for st, val in states.items():
            pool = [q for q in st if q[1] >= t] + list(arr[t])
            options = [None] + list(range(len(pool)))
            for i in options:
                gain, rest = 0.0, list(pool)
                if i is not None:
                    gain = pool[i][0]
                    rest.pop(i)
                keep = sorted([q for q in rest if q[1] > t], key=lambda q: -q[0])[:keep_n]
                key = tuple(sorted(keep))
                v = val + gain
                if nxt.get(key, -1.0) < v:
                    nxt[key] = v
        states = nxt
    return max(states.values())


def exp3_alpha_regret(T, K, seed, alpha, arms, sbound=2, block=25):
    """EXP3 over a finite set of theta-arms.

    Returns (learning_regret, alg_total, opt_total, alpha_slack) where
      learning_regret = best-fixed-arm total  -  learner total   (the O~(sqrt(KT)) term)
      alpha_slack     = alg_total - alpha*opt_total  (>=0 means the alpha-competitive
                        benchmark part of the alpha-regret bound is met)."""
    rng = np.random.default_rng(seed)
    values = np.linspace(1.0, 2.0, K)
    n = len(arms)
    w = np.ones(n)
    n_blocks = max(1, T // block)
    eta = np.sqrt(np.log(n) / (n_blocks * n))
    alg_tot = opt_tot = 0.0
    arm_tot = np.zeros(n)
    for _ in range(n_blocks):
        p = w / w.sum()
        i = int(rng.choice(n, p=p))
        arr = sample_instance(rng, block, K, values, sbound)
        vals_all = np.array([run_alg_theta(arr, block, th) for th in arms])
        arm_tot += vals_all
        a = vals_all[i]
        o = opt_dp(arr, block, sbound)
        alg_tot += a
        opt_tot += o
        w[i] *= np.exp(eta * (a / max(1.0, o)) / p[i])
        w /= w.max()
    learn_regret = float(arm_tot.max() - alg_tot)
    return learn_regret, alg_tot, opt_tot, float(alg_tot - alpha * opt_tot)


def fit_scaling(grid, Ks, Ts):
    X = np.array([[1.0, np.log(T), np.log(K)] for K in Ks for T in Ts])
    y = np.array([np.log(max(grid[f"K{K}_T{T}"], 1e-6)) for K in Ks for T in Ts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"const": float(coef[0]), "exp_T": float(coef[1]), "exp_K": float(coef[2])}
