"""Randomized-algorithm competitive-ratio machinery shared by claims 4 and 5.

ALG^R (mixing rule): at each step let e = best packet expiring now, h = best
pending packet. With probability p send h, with probability 1-p send e
(if nothing expires, send h). Expectation is computed EXACTLY by recursing over
the induced distribution of pending sets (no Monte-Carlo noise).
"""
from functools import lru_cache
from common import opt_value, enumerate_instances


def expected_alg(arrivals, T, p):
    def rec(t, pending, prob):
        if prob < 1e-15:
            return 0.0
        if t == T:
            return 0.0
        pend = [q for q in pending if q[1] >= t] + list(arrivals[t])
        if not pend:
            return rec(t + 1, (), prob)
        exp_now = [q for q in pend if q[1] == t]
        h = max(pend, key=lambda q: q[0])
        val = 0.0
        if exp_now:
            e = max(exp_now, key=lambda q: q[0])
            if e is h or e == h:
                rest = list(pend); rest.remove(h)
                return h[0] * prob + rec(t + 1, tuple(rest), prob)
            for q, w in ((h, p), (e, 1 - p)):
                if w <= 0:
                    continue
                rest = list(pend); rest.remove(q)
                val += q[0] * prob * w + rec(t + 1, tuple(rest), prob * w)
        else:
            rest = list(pend); rest.remove(h)
            val += h[0] * prob + rec(t + 1, tuple(rest), prob)
        return val

    return rec(0, (), 1.0)


def worst_ratio_randomized(values, p, T, max_arrivals=2, sbound=2):
    worst, arg = 1.0, None
    for arr in enumerate_instances(values, T, max_arrivals, sbound):
        o = opt_value(arr, T)
        if o <= 0:
            continue
        a = expected_alg(arr, T, p)
        if a <= 1e-12:
            return float("inf"), arr
        r = o / a
        if r > worst + 1e-12:
            worst, arg = r, arr
    return worst, arg
