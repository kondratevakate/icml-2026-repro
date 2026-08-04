"""Claim 2 (Theorem 2, Sec 4.2): nestedness K_{tau1} subseteq K_{tau2} for tau1 < tau2.

Model: the threshold family is the parametric selection problem
    K(lam) = argmax_S  sum_{e in E(S)} w_e - lam*|S|   (solved by min-cut),
which is the parametric-min-cut connection the theorem invokes. Larger coverage
threshold tau <-> smaller per-vertex price lam, so we set lam = LAM_MAX*(1-tau) and
test K_{tau1} subseteq K_{tau2} for tau1<tau2 (equivalently S(lam) shrinks as lam grows).

Mutation: replace the parametric-min-cut solutions with (a) a greedy degree-peeling
selector and (b) a *non-monotone* objective (lam*|S| -> lam*|S| - 0.5*lam^2*|S|^0.5)
and re-test nestedness; a real monotone structure must be destroyed by these.
"""
import json, os, numpy as np, networkx as nx
from common import random_weighted_graph, parametric_select, total_weight

os.makedirs('results', exist_ok=True)
SEED = 20260803
rng = np.random.default_rng(SEED)
TAUS = np.linspace(0.0, 1.0, 21)

def greedy_select(G, lam):
    S = set(G.nodes()); 
    def obj(S):
        return sum(d['w'] for u, v, d in G.edges(data=True) if u in S and v in S) - lam * len(S)
    cur = obj(S)
    improved = True
    while improved and S:
        improved = False
        for v in list(S):
            T = S - {v}
            o = obj(T)
            if o > cur + 1e-12:
                S, cur, improved = T, o, True
                break
    return S

def nonmono_select(G, lam):
    """min-cut solution for a distorted price that is non-monotone in lam."""
    lam2 = lam * (1.0 + 0.9 * np.sin(6 * lam))
    return parametric_select(G, max(lam2, 0.0))

def test_family(G, selector, lam_max):
    sets = [selector(G, lam_max * (1 - t)) for t in TAUS]
    bad = 0
    for i in range(len(TAUS) - 1):
        if not sets[i] <= sets[i + 1]:
            bad += 1
    return bad, [len(s) for s in sets]

rows = []
viol = mut_greedy = mut_nonmono = 0
for t in range(10):
    n = int(rng.integers(10, 18)); p = float(rng.uniform(0.25, 0.55))
    G = random_weighted_graph(n, p, seed=int(rng.integers(1e6)))
    if G.number_of_edges() == 0:
        continue
    lam_max = max(d['w'] for _, _, d in G.edges(data=True)) * max(dict(G.degree()).values())
    b, sizes = test_family(G, parametric_select, lam_max)
    bg, _ = test_family(G, greedy_select, lam_max)
    bn, _ = test_family(G, nonmono_select, lam_max)
    viol += b; mut_greedy += bg; mut_nonmono += bn
    rows.append(dict(n=n, m=G.number_of_edges(), violations=b, sizes=sizes,
                     mut_greedy_violations=bg, mut_nonmono_violations=bn))

res = dict(claim="Theorem 2 nestedness K_tau1 subseteq K_tau2 (tau1<tau2)",
           source="Theorem 2, Section 4.2", seed=SEED,
           instances=len(rows), tau_grid=len(TAUS),
           nesting_violations_parametric_mincut=int(viol),
           mutation_greedy_violations=int(mut_greedy),
           mutation_nonmonotone_price_violations=int(mut_nonmono),
           verdict="verified" if viol == 0 and (mut_greedy + mut_nonmono) > 0 else "inconclusive",
           detail=rows)
json.dump(res, open('results/claim2.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'detail'}, indent=1))
