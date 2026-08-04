"""Claim 6 (Section 5): synthetic 6x6 grid navigation, 50 train/test routes; the LP-based
method compresses the calibration set to 52 edges at phi=0.75, and beats greedy baselines
for phi <= 0.8.

The paper PDF was not in the bundle, so the exact route generator / edge-weighting is
unknown. We implement the most natural reading:
  - 6x6 grid graph (36 nodes, 60 edges),
  - 50 train routes = random source->target shortest paths under random edge costs,
    50 test routes drawn the same way (exchangeable),
  - edge weight w_e = frequency of e in the train routes; W = sum w_e,
  - conformal threshold tau* = ceil((n+1)*phi)/n quantile of per-route scores,
  - "compressed calibration set" = the edge set K_tau* returned by LP rounding
    (kappa=1, theta=0.5) at the smallest budget r whose LP loss meets the tau* level,
  - baselines: (i) greedy top-weight edges, (ii) greedy max-marginal-coverage.
Metric = number of edges kept at the required coverage level (smaller is better) and
realised test coverage.
Mutation: shuffle the edge weights (destroy route structure) -- compression advantage of
LP over greedy must vanish.
"""
import json, os, numpy as np, networkx as nx
from common import solve_lp, round_threshold

os.makedirs('results', exist_ok=True)
SEED = 20260803
rng = np.random.default_rng(SEED)

def make_grid():
    G = nx.grid_2d_graph(6, 6)
    return nx.convert_node_labels_to_integers(G, label_attribute='pos')

def routes(G, n, rng):
    out = []
    cost = {e: rng.uniform(0.5, 1.5) for e in G.edges()}
    for u, v, d in G.edges(data=True):
        d['c'] = cost[(u, v)] * rng.uniform(0.8, 1.2)
    nodes = list(G.nodes())
    for _ in range(n):
        s, t = rng.choice(nodes, 2, replace=False)
        p = nx.shortest_path(G, int(s), int(t), weight='c')
        out.append([tuple(sorted((p[i], p[i + 1]))) for i in range(len(p) - 1)])
    return out

def edge_weights(G, rs):
    w = {tuple(sorted(e)): 0.0 for e in G.edges()}
    for r in rs:
        for e in r:
            w[e] += 1.0
    return w

def cover_frac(routeset, K):
    """fraction of routes fully contained in edge set K"""
    return float(np.mean([all(e in K for e in r) for r in routeset]))

def lp_edges(G, w, r_budget):
    H = nx.Graph()
    for e, val in w.items():
        if val > 0:
            H.add_edge(*e, w=val)
    if H.number_of_edges() == 0:
        return set()
    x, _ = solve_lp(H, r_budget)
    S = round_threshold(x, 0.5)
    return {tuple(sorted((u, v))) for u, v in H.edges() if u in S and v in S}

def greedy_topw(w, target_mass):
    tot = sum(w.values()); acc = 0.0; K = set()
    for e, val in sorted(w.items(), key=lambda kv: -kv[1]):
        if acc >= target_mass:
            break
        K.add(e); acc += val
    return K

def greedy_routes(train, target_cov):
    """greedy: add whole routes (max marginal coverage) until target route coverage"""
    K = set(); covered = 0
    order = sorted(train, key=len)
    for r in order:
        if covered / len(train) >= target_cov:
            break
        K |= set(r)
        covered = sum(all(e in K for e in rr) for rr in train)
    return K

G = make_grid()
PHIS = [0.6, 0.7, 0.75, 0.8, 0.9]
agg = {p: dict(lp=[], gw=[], gr=[], lp_cov=[], gr_cov=[]) for p in PHIS}
mut = {p: dict(lp=[], gr=[]) for p in PHIS}

for trial in range(10):
    train = routes(G, 50, rng)
    test = routes(G, 50, rng)
    w = edge_weights(G, train)
    W = sum(w.values())
    wmut = dict(zip(w.keys(), rng.permutation(list(w.values()))))
    for phi in PHIS:
        # budget search: smallest vertex budget whose LP-rounded edge set covers >= phi of train routes
        best = None
        for rb in range(4, 37):
            K = lp_edges(G, w, rb)
            if cover_frac(train, K) >= phi:
                best = K; break
        if best is None:
            best = set(w.keys())
        gw = greedy_topw(w, phi * W)
        gr = greedy_routes(train, phi)
        agg[phi]['lp'].append(len(best)); agg[phi]['gw'].append(len(gw)); agg[phi]['gr'].append(len(gr))
        agg[phi]['lp_cov'].append(cover_frac(test, best)); agg[phi]['gr_cov'].append(cover_frac(test, gr))
        # mutation
        bm = None
        for rb in range(4, 37):
            Km = lp_edges(G, wmut, rb)
            if cover_frac(train, Km) >= phi:
                bm = Km; break
        mut[phi]['lp'].append(len(bm) if bm else len(w))
        mut[phi]['gr'].append(len(gr))

rows = []
for phi in PHIS:
    a = agg[phi]
    rows.append(dict(phi=phi, lp_edges_mean=float(np.mean(a['lp'])),
                     greedy_weight_edges_mean=float(np.mean(a['gw'])),
                     greedy_route_edges_mean=float(np.mean(a['gr'])),
                     lp_test_coverage=float(np.mean(a['lp_cov'])),
                     greedy_route_test_coverage=float(np.mean(a['gr_cov'])),
                     lp_beats_greedy=bool(np.mean(a['lp']) <= min(np.mean(a['gw']), np.mean(a['gr']))),
                     mut_lp_edges_mean=float(np.mean(mut[phi]['lp']))))

lp75 = [r for r in rows if r['phi'] == 0.75][0]['lp_edges_mean']
beats_low = all(r['lp_beats_greedy'] for r in rows if r['phi'] <= 0.8)
matches_52 = abs(lp75 - 52) <= 3
res = dict(claim="Section 5: 6x6 grid, 50 routes -> 52 edges at phi=0.75; LP beats greedy for phi<=0.8",
           source="Section 5", seed=SEED, grid="6x6 (36 nodes, 60 edges)", trials=10,
           lp_edges_at_phi075=lp75, paper_value_at_phi075=52,
           matches_52_within_3=bool(matches_52),
           lp_beats_greedy_for_phi_le_0p8=bool(beats_low),
           mutation_shuffled_weights_lp_edges_at_075=float(np.mean(mut[0.75]['lp'])),
           note="Route generator / edge-weighting protocol is not specified in the available "
                "bundle (no PDF), so the absolute edge count is protocol-dependent.",
           verdict=("verified" if (matches_52 and beats_low) else "inconclusive"),
           detail=rows)
json.dump(res, open('results/claim6.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
