"""Claim 6 (Lemma 2.1): sleeping bandits are a special case of 1-bounded K-OPSD.

Constructive check of the reduction. Given a sleeping-bandit instance
(K arms, per-round availability set A_t, rewards r_{t,k}) build the 1-bounded
K-OPSD instance: at round t, for each k in A_t a packet of type k with value
r_{t,k}, release t, deadline t (1-bounded => must be sent on arrival or lost);
one transmission per step == one arm pull per round.

We verify the reduction is reward-preserving *pathwise* for arbitrary policies
(random, greedy, worst) and that the offline optima coincide.
Seed: 20260803.
"""
import json
import numpy as np
from common import opt_value

SEED = 20260803


def sleeping_bandit(rng, K, T, p_avail=0.6):
    avail = [np.where(rng.random(K) < p_avail)[0].tolist() for _ in range(T)]
    for t in range(T):
        if not avail[t]:
            avail[t] = [int(rng.integers(K))]
    rew = rng.random((T, K))
    return avail, rew


def to_opsd(avail, rew):
    return [[(float(rew[t, k]), t) for k in avail[t]] for t in range(len(avail))]


def run_policy(avail, rew, choices):
    """Sleeping bandit: pull choices[t] (must be available); collect reward."""
    return sum(float(rew[t, choices[t]]) for t in range(len(avail)))


def run_policy_opsd(arrivals, choices, avail):
    """Same choices executed in the OPSD instance (send the packet of that type)."""
    tot = 0.0
    for t, step in enumerate(arrivals):
        k = choices[t]
        idx = avail[t].index(k)
        tot += step[idx][0]
    return tot


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    diffs, opt_diffs = [], []
    for trial in range(200):
        K = int(rng.integers(2, 6))
        T = int(rng.integers(2, 7))
        avail, rew = sleeping_bandit(rng, K, T)
        arr = to_opsd(avail, rew)
        # random policy
        ch = [int(avail[t][rng.integers(len(avail[t]))]) for t in range(T)]
        diffs.append(abs(run_policy(avail, rew, ch) - run_policy_opsd(arr, ch, avail)))
        # greedy policy
        chg = [int(max(avail[t], key=lambda k: rew[t, k])) for t in range(T)]
        diffs.append(abs(run_policy(avail, rew, chg) - run_policy_opsd(arr, chg, avail)))
        # offline optima
        sb_opt = sum(max(rew[t, k] for k in avail[t]) for t in range(T))
        opt_diffs.append(abs(sb_opt - opt_value(arr, T)))

    # MUTATION: break 1-boundedness (give packets deadline t+1, i.e. 2-bounded).
    # Then the OPSD optimum may strictly exceed the sleeping-bandit optimum,
    # so the reduction must fail.
    rng2 = np.random.default_rng(SEED + 1)
    mut_gap = []
    for _ in range(60):
        K, T = 3, 4
        avail, rew = sleeping_bandit(rng2, K, T)
        arr2 = [[(float(rew[t, k]), min(t + 1, T - 1)) for k in avail[t]] for t in range(T)]
        sb_opt = sum(max(rew[t, k] for k in avail[t]) for t in range(T))
        mut_gap.append(opt_value(arr2, T) - sb_opt)

    out = {
        "claim": 6, "source": "Lemma 2.1 / Section 2.2", "seed": SEED,
        "n_trials": 200,
        "max_abs_pathwise_reward_diff": float(np.max(diffs)),
        "max_abs_offline_opt_diff": float(np.max(opt_diffs)),
        "reduction_exact": bool(np.max(diffs) < 1e-12 and np.max(opt_diffs) < 1e-9),
        "mutation_2bounded_mean_opt_gap": float(np.mean(mut_gap)),
        "mutation_2bounded_frac_strictly_better": float(np.mean(np.array(mut_gap) > 1e-9)),
    }
    json.dump(out, open("results/claim6.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
