"""Claim 4 (Theorem 5.1): randomized ALG^R2 on 2-bounded instances -- the benchmark
constant is 5/4, i.e. the randomized algorithm is 5/4-competitive and its
5/4-regret is O~(sqrt(KT)).

Part A (competitive constant, exhaustive): compute the exact worst-case
E[OPT/ALG] of the randomized mixing algorithm over ALL 2-bounded instances of
length T with values from a K-type set, optimized over the mixing probability p.
Claim predicts min_p worst-case ratio = 1.25.

Part B (regret scaling): stochastic 2-bounded environment, EXP3-style learner over
a grid of mixing probabilities; measure the 5/4-regret  R = (4/5)OPT_T - ALG_T
and fit R ~ C * T^a * K^b in log-log.
Seeds: 20260803 + trial index.
"""
import json, time
import numpy as np
from common import opt_value
from randomized import worst_ratio_randomized
from learning import exp3_alpha_regret

SEED = 20260803


# ---------------- Part B environment ----------------
def sample_instance(rng, T, K, values):
    """i.i.d. stochastic 2-bounded arrivals: per step, 0-2 packets."""
    arr = []
    for t in range(T):
        n = int(rng.integers(0, 3))
        step = []
        for _ in range(n):
            v = float(values[rng.integers(K)])
            d = t + int(rng.integers(0, 2))
            step.append((v, min(d, T - 1)))
        arr.append(step)
    return arr


def run_alg(arr, T, p, rng):
    pending, tot = [], 0.0
    for t in range(T):
        pending = [q for q in pending if q[1] >= t] + list(arr[t])
        if not pending:
            continue
        exp_now = [q for q in pending if q[1] == t]
        h = max(pending, key=lambda q: q[0])
        if exp_now:
            e = max(exp_now, key=lambda q: q[0])
            send = h if rng.random() < p else e
        else:
            send = h
        tot += send[0]
        pending.remove(send)
    return tot


def opt_greedy(arr, T):
    """Exact offline optimum for 2-bounded by DP on (t, best carried packet)."""
    carry = {None: 0.0}
    for t in range(T):
        nxt = {}
        for c, val in carry.items():
            pool = ([c] if c is not None else []) + [q for q in arr[t]]
            pool = [q for q in pool if q[1] >= t]
            cands = [None] + list(range(len(pool)))
            for i in cands:
                gain = 0.0
                rest = list(pool)
                if i is not None:
                    gain = pool[i][0]
                    rest.pop(i)
                keep = [q for q in rest if q[1] > t]
                nc = max(keep, key=lambda q: q[0]) if keep else None
                v = val + gain
                if nxt.get(nc, -1) < v:
                    nxt[nc] = v
        carry = nxt
    return max(carry.values())


def exp3_regret(T, K, seed, alpha=0.8, n_p=9):
    """Learner: EXP3 over mixing probabilities p in [0,1] (bandit feedback per block)."""
    rng = np.random.default_rng(seed)
    values = np.linspace(1.0, 2.0, K)
    ps = np.linspace(0.0, 1.0, n_p)
    w = np.ones(n_p)
    block = 25
    n_blocks = max(1, T // block)
    alg_tot, opt_tot = 0.0, 0.0
    eta = np.sqrt(np.log(n_p) / (n_blocks * n_p))
    for b in range(n_blocks):
        prob = w / w.sum()
        i = int(rng.choice(n_p, p=prob))
        arr = sample_instance(rng, block, K, values)
        a = run_alg(arr, block, ps[i], rng)
        o = opt_greedy(arr, block)
        alg_tot += a
        opt_tot += o
        scale = max(1.0, o)
        w[i] *= np.exp(eta * (a / scale) / prob[i])
        w /= w.max()
    return alpha * opt_tot - alg_tot, alg_tot, opt_tot


if __name__ == "__main__":
    t0 = time.time()
    out = {"claim": 4, "source": "Theorem 5.1", "seed": SEED, "target_constant": 1.25}

    # Part A
    partA = {}
    for K, vals in [(2, [1.0, 1.6]), (3, [1.0, 1.4, 2.0])]:
        best = (None, 1e9)
        curve = {}
        for p in np.round(np.arange(0.0, 1.01, 0.1), 2):
            r, _ = worst_ratio_randomized(vals, float(p), 3)
            curve[str(p)] = r
            if r < best[1]:
                best = (float(p), r)
        partA[str(K)] = {"values": vals, "best_p": best[0],
                         "worst_case_ratio": best[1], "curve": curve}
        print("A", K, best, f"{time.time()-t0:.1f}s", flush=True)
    out["part_A_competitive"] = partA
    ratios = [partA[k]["worst_case_ratio"] for k in partA]
    out["A_ratio_le_5_4"] = all(r <= 1.25 + 1e-6 for r in ratios)
    out["A_min_ratio"] = min(ratios)
    out["A_max_ratio"] = max(ratios)
    # mutation: deterministic p=1 (always greedy-on-h) must be strictly worse
    out["A_mutation_p1_ratio"] = {k: partA[k]["curve"]["1.0"] for k in partA}
    out["A_mutation_worse"] = all(partA[k]["curve"]["1.0"] > partA[k]["worst_case_ratio"] + 1e-9
                                  for k in partA)

    # Part B: scaling of 5/4-regret
    Ts = [500, 1000, 2000, 4000, 8000]
    Ks = [2, 4, 8, 16]
    grid = {}
    for K in Ks:
        for T in Ts:
            rs = [exp3_alpha_regret(T, K, SEED + 1000 * K + s, 0.8,
                                    [1.0, 1.2, 1.4, 1.6, 1.8, 2.0], 2)[0] for s in range(5)]
            grid[f"K{K}_T{T}"] = float(np.mean(rs))
    out["part_B_regret_grid"] = grid
    X = np.array([[1.0, np.log(T), np.log(K)] for K in Ks for T in Ts])
    y = np.array([np.log(max(grid[f"K{K}_T{T}"], 1e-6)) for K in Ks for T in Ts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    out["fit_log_regret"] = {"const": float(coef[0]), "exp_T": float(coef[1]),
                             "exp_K": float(coef[2])}
    out["exp_T_near_half"] = bool(abs(coef[1] - 0.5) < 0.15)
    out["exp_K_near_half"] = bool(abs(coef[2] - 0.5) < 0.25)
    slack = [exp3_alpha_regret(2000, K, SEED + K, 0.8, [1.0, 1.2, 1.4, 1.6, 1.8, 2.0], 2)[3]
             for K in Ks]
    out["alpha_slack_T2000"] = [float(x) for x in slack]
    out["alpha_benchmark_met"] = bool(all(x >= 0 for x in slack))
    out["runtime_s"] = time.time() - t0
    json.dump(out, open("results/claim4.json", "w"), indent=2)
    print(json.dumps(out, indent=2)[:2500])
