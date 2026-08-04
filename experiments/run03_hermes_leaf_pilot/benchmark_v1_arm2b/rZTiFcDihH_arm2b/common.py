"""Shared first-principles model for K-OPSD (Online Packet Scheduling with Deadlines).

Model (as stated in the anchored claims; paper PDF was NOT retrievable, see logbook):
  - Time steps t = 1..T. At each step a set of packets arrives.
  - A packet = (value v from a finite set of K "types", release r, deadline d).
  - s-bounded: d - r <= s - 1  (2-bounded: deadline is r or r+1).
  - One packet may be transmitted per step; value of transmitted packets is collected.
  - OPT = offline optimum (max-weight schedule).
"""
import itertools
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def opt_value(arrivals, T):
    """Offline optimum by DP over (t, set of still-pending packets).
    arrivals: list of length T; arrivals[t] = list of (value, deadline).
    2-/3-bounded, small instances -> brute force over schedules."""
    best = [0.0]

    def rec(t, pending, acc):
        if t == T:
            best[0] = max(best[0], acc)
            return
        pend = [p for p in pending if p[1] >= t] + list(arrivals[t])
        # option: idle
        rec(t + 1, tuple(p for p in pend if p[1] > t), acc)
        seen = set()
        for i, p in enumerate(pend):
            if p in seen:
                continue
            seen.add(p)
            rest = pend[:i] + pend[i + 1:]
            rec(t + 1, tuple(q for q in rest if q[1] > t), acc + p[0])

    rec(0, (), 0.0)
    return best[0]


def alg_theta(arrivals, T, theta):
    """Deterministic ALG^theta for bounded-deadline instances.
    At each step: e = max-value packet expiring now, h = max-value pending packet.
    Send h if h >= theta * e, else send e.  (theta = PHI recovers the classical
    golden-ratio algorithm.)"""
    pending = []
    total = 0.0
    for t in range(T):
        pending = [p for p in pending if p[1] >= t] + list(arrivals[t])
        if not pending:
            continue
        exp_now = [p for p in pending if p[1] == t]
        h = max(pending, key=lambda p: p[0])
        if exp_now:
            e = max(exp_now, key=lambda p: p[0])
            send = h if h[0] >= theta * e[0] else e
        else:
            send = h
        total += send[0]
        pending.remove(send)
    return total


def enumerate_instances(values, T, max_arrivals=2, sbound=2):
    """All arrival sequences: each step, a multiset of <=max_arrivals packets,
    each with value in `values` and deadline in {t, ..., t+sbound-1}."""
    per_step = []
    single = [(v, off) for v in values for off in range(sbound)]
    opts = [()]
    for k in range(1, max_arrivals + 1):
        opts += list(itertools.combinations_with_replacement(single, k))
    for _ in range(T):
        per_step.append(opts)
    for seq in itertools.product(*per_step):
        yield [[(v, t + off) for (v, off) in step] for t, step in enumerate(seq)]


def worst_case_ratio(values, theta, T, max_arrivals=2, sbound=2):
    worst = 1.0
    arg = None
    for arr in enumerate_instances(values, T, max_arrivals, sbound):
        a = alg_theta(arr, T, theta)
        if a <= 0:
            o = opt_value(arr, T)
            if o > 0:
                return float("inf"), arr
            continue
        o = opt_value(arr, T)
        r = o / a
        if r > worst + 1e-12:
            worst, arg = r, arr
    return worst, arg
