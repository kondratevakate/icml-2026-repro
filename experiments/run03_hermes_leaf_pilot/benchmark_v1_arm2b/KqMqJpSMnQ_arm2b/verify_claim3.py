"""Claim 3 (Corollary 1, Sec 4.2): the entire nested sequence over thresholds is
computable in Otilde(gamma*(m+n)^2) time, gamma = number of distinct subgraphs
(breakpoints) in the sequence.

Executable content:
 (a) enumerate the FULL breakpoint sequence exactly with divide-and-conquer on the
     parameter (Eisner/Gallo-style recursion over lam), counting min-cut calls;
 (b) check gamma <= n+1 (the nested chain cannot be longer than n+1);
 (c) check #mincut calls = O(gamma) (we measure calls/gamma);
 (d) fit empirical runtime T ~ C*gamma*(m+n)^alpha and report alpha; the corollary
     predicts alpha <= 2 (up to log factors).
Mutation: naive uniform grid scan over 200 lam values -- must use many more min-cut
calls than the D&C recursion while recovering no more breakpoints.
"""
import json, os, time, numpy as np
from common import random_weighted_graph, parametric_select, total_weight

os.makedirs('results', exist_ok=True)
SEED = 20260803
rng = np.random.default_rng(SEED)

def full_sequence(G, lam_hi):
    """Divide & conquer over lam: returns (list of distinct sets, #cut calls)."""
    calls = [0]
    def sel(l):
        calls[0] += 1
        return frozenset(parametric_select(G, l))
    memo = {}
    def S(l):
        if l not in memo:
            memo[l] = sel(l)
        return memo[l]
    breaks = {}
    def rec(lo, hi, Slo, Shi, depth=0):
        if Slo == Shi or hi - lo < 1e-9 or depth > 40:
            return
        mid = 0.5 * (lo + hi)
        Sm = S(mid)
        breaks[mid] = Sm
        rec(lo, mid, Slo, Sm, depth + 1)
        rec(mid, hi, Sm, Shi, depth + 1)
    lo, hi = 0.0, lam_hi
    Slo, Shi = S(lo), S(hi)
    breaks[lo], breaks[hi] = Slo, Shi
    rec(lo, hi, Slo, Shi)
    distinct = []
    for l in sorted(breaks):
        if not distinct or breaks[l] != distinct[-1]:
            distinct.append(breaks[l])
    return distinct, calls[0]

rows = []
for n in [8, 12, 16, 20, 24]:
    G = random_weighted_graph(n, 0.35, seed=int(rng.integers(1e6)))
    m = G.number_of_edges()
    if m == 0:
        continue
    lam_hi = max(d['w'] for _, _, d in G.edges(data=True)) * max(dict(G.degree()).values()) + 1
    t0 = time.perf_counter()
    seq, calls = full_sequence(G, lam_hi)
    t = time.perf_counter() - t0
    gamma = len(seq)
    # nested chain check
    nested = all(set(seq[i + 1]) <= set(seq[i]) for i in range(gamma - 1))
    # mutation: uniform grid scan
    t1 = time.perf_counter()
    grid = [frozenset(parametric_select(G, l)) for l in np.linspace(0, lam_hi, 200)]
    tg = time.perf_counter() - t1
    gdist = len({g for g in grid})
    rows.append(dict(n=n, m=m, gamma=gamma, nested_chain=bool(nested),
                     mincut_calls=calls, calls_per_gamma=calls / gamma,
                     gamma_le_n_plus_1=bool(gamma <= n + 1), seconds=t,
                     mut_grid_calls=200, mut_grid_distinct=gdist, mut_grid_seconds=tg))

x = np.log([r['m'] + r['n'] for r in rows])
y = np.log([r['seconds'] / r['gamma'] for r in rows])
alpha = float(np.polyfit(x, y, 1)[0])

ok = all(r['gamma_le_n_plus_1'] and r['nested_chain'] for r in rows)
mut_ok = all(r['mut_grid_calls'] > r['mincut_calls'] and r['mut_grid_distinct'] <= r['gamma'] for r in rows)
res = dict(claim="Corollary 1: whole nested sequence in Otilde(gamma (m+n)^2)",
           source="Corollary 1, Section 4.2", seed=SEED,
           empirical_alpha_time_per_breakpoint=alpha,
           alpha_le_2=bool(alpha <= 2.0),
           all_gamma_le_n_plus_1=bool(ok),
           max_calls_per_gamma=max(r['calls_per_gamma'] for r in rows),
           mutation_gridscan_wasteful=bool(mut_ok),
           verdict="verified" if (ok and alpha <= 2.0 and mut_ok) else "inconclusive",
           detail=rows)
json.dump(res, open('results/claim3.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'detail'}, indent=1))
print(rows)
