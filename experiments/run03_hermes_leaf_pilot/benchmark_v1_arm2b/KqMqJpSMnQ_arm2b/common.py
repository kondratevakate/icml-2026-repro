"""Shared first-principles model for 'Compact Conformal Subgraphs' (orid KqMqJpSMnQ).

Paper PDF was NOT available in the bundle; the model below is reconstructed from the
anchored claim statements so that all six claims are mutually consistent:

  Graph G=(V,E), edge weights w_e>0, W = sum_e w_e.
  A "subgraph" K is a vertex set; its coverage is e(K) = sum of w_e over edges
  induced by K.  Coverage loss = W - e(K).  Size = |K|.
  Primal problem P(eps): minimize |K| s.t. W - e(K) <= eps*W.
  Budgeted form   B(r): maximize e(K) s.t. |K| <= r.   (densest-r-subgraph form,
  which is exactly what makes Theorem 4's clique reduction go through.)

LP relaxation (Section 4.1 style):
  max sum_e w_e y_e  s.t. y_e <= x_u, y_e <= x_v, sum_v x_v <= r, 0<=x,y<=1.
Threshold rounding at theta = kappa/(1+kappa):  K = {v : x_v >= theta}.
"""
import numpy as np, networkx as nx
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD, value


def random_weighted_graph(n, p, seed, wmax=1.0):
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(n, p, seed=int(seed))
    for u, v in G.edges():
        G[u][v]['w'] = float(rng.uniform(0.1, wmax))
    return G


def coverage(G, S):
    S = set(S)
    return sum(d['w'] for u, v, d in G.edges(data=True) if u in S and v in S)


def total_weight(G):
    return sum(d['w'] for _, _, d in G.edges(data=True))


def solve_lp(G, r):
    """Fractional optimum of B(r). Returns (x dict, lp_value)."""
    prob = LpProblem("dks", LpMaximize)
    x = {v: LpVariable(f"x{v}", 0, 1) for v in G.nodes()}
    y = {}
    for u, v, d in G.edges(data=True):
        y[(u, v)] = LpVariable(f"y{u}_{v}", 0, 1)
    prob += lpSum(G[u][v]['w'] * y[(u, v)] for u, v in y)
    for (u, v) in y:
        prob += y[(u, v)] <= x[u]
        prob += y[(u, v)] <= x[v]
    prob += lpSum(x.values()) <= r
    prob.solve(PULP_CBC_CMD(msg=0))
    return {v: x[v].value() for v in x}, value(prob.objective)


def round_threshold(xsol, theta):
    return {v for v, xv in xsol.items() if xv >= theta - 1e-9}


def brute_force_best(G, r):
    """Exact max coverage with |K|<=r (small n only)."""
    from itertools import combinations
    nodes = list(G.nodes())
    best, bestS = -1.0, set()
    for k in range(1, r + 1):
        for S in combinations(nodes, k):
            c = coverage(G, S)
            if c > best:
                best, bestS = c, set(S)
    return best, bestS


# ---- parametric selection / min-cut machinery (Theorem 2, Corollary 1) ----
def parametric_select(G, lam):
    """Maximize sum_{e in E(S)} w_e - lam*|S| by min-cut on the project-selection net.
    Returns the *minimal* maximizer (source side reachable in residual graph)."""
    D = nx.DiGraph()
    INF = float('inf')
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        en = ('e', i)
        D.add_edge('s', en, capacity=d['w'])
        D.add_edge(en, ('v', u), capacity=INF)
        D.add_edge(en, ('v', v), capacity=INF)
    for v in G.nodes():
        D.add_edge(('v', v), 't', capacity=max(lam, 0.0))
    cut, (src, _) = nx.minimum_cut(D, 's', 't')
    return {v for kind, v in (x for x in src if isinstance(x, tuple)) if kind == 'v'}
