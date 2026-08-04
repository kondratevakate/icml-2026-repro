"""CLAIM 4 - Lemma 1 (Sec.2.1, proof App.A): distribution-free marginal coverage
P(B* in K_{tau*}(A*)) >= phi - delta under exchangeability (no model probability estimates needed).

Reproduction (fixed-context split-conformal; faithfully mirrors App.A's proof structure):
  * A grid; routes are shortest paths under random edge weights (exchangeable i.i.d. draws).
  * With probability delta, a "wild" route uses an edge outside the grid -> can never be covered
    (this is exactly the Stage-1 failure B* notin S_{d*} that costs the delta in Lemma 1).
  * Build the route hypergraph; compute the nested conformal-subgraph sequence {K_tau} (parametric
    min-cut). For each calibration route, its nested nonconformity score
    eta = min{tau : route subseteq K_tau} (1 if it uses an outside edge).
  * Calibrate tau* as the ceil(phi*(N+1))-th smallest eta on the calibration split (split-conformal
    quantile). Test coverage = fraction of test routes with eta* <= tau*  <=>  route subseteq K_{tau*}.
  * Repeat over many exchangeable trials; estimate P(coverage) and check >= phi - delta.
Mutation: under-calibrate tau* (wrong, too-small quantile) -> empirical coverage drops below phi-delta,
showing the guarantee is real and hinges on correct split-conformal calibration (the core of Lemma 1).
"""
import json
import heapq
import numpy as np
from lib_flow import parametric_sequence

SEED = 20260207


def build_grid(g):
    nodes = [(i, j) for i in range(g) for j in range(g)]
    nid = {u: k for k, u in enumerate(nodes)}
    adj = {u: [] for u in nodes}
    edges = {}
    eid = 0
    for (i, j) in nodes:
        for (di, dj) in [(1, 0), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < g and 0 <= nj < g:
                u, v = (i, j), (ni, nj)
                adj[u].append(v); adj[v].append(u)
                edges[(nid[u], nid[v])] = eid; eid += 1
    return nid, adj, eid, edges  # eid = number of grid edges


def shortest_path_edges(s, t, adj, nid, weights, edges):
    # Dijkstra on node graph with edge weights; returns list of grid-edge indices on the path
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        if u == t:
            break
        for v in adj[u]:
            w = weights[tuple(sorted((nid[u], nid[v])))]
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    path = []
    cur = t
    while cur != s:
        p = prev[cur]
        path.append(edges[tuple(sorted((nid[p], nid[cur])))])
        cur = p
    return path


def gen_routes(g, n_routes, delta, seed, bypass_edges=None):
    rng = np.random.default_rng(seed)
    nid, adj, E, edges = build_grid(g)
    s, t = (0, 0), (g - 1, g - 1)
    weights = {}
    for u in adj:
        for v in adj[u]:
            key = tuple(sorted((nid[u], nid[v])))
            if key not in weights:
                weights[key] = rng.uniform(0.1, 2.0)
    routes = []
    for _ in range(n_routes):
        wild = rng.random() < delta
        if wild:
            path = shortest_path_edges(s, t, adj, nid, weights, edges)
            route = tuple(sorted(set(path) | {E}))  # outside edge index E -> never covered
        else:
            path = shortest_path_edges(s, t, adj, nid, weights, edges)
            route = tuple(sorted(set(path)))
        routes.append(route)
    n_vertices = E + 1
    return routes, n_vertices


def nested_seq_from_routes(routes, n_vertices, lam_grid):
    # vertices are edge indices 0..n_vertices-1; hyperedges are routes
    vertices = list(range(n_vertices))
    weights = [1.0 / len(routes)] * len(routes)
    seq = parametric_sequence(vertices, routes, weights, lam_grid)
    return seq


def compute_eta_scores(routes, seq):
    # eta_i = first sequence level that contains route i; coverage level = fraction of all
    # hyperedges contained in that level (i.e., e(S_j)/W)
    total = len(routes)
    # precompute e(S_j)/W for each level
    level_cov = []
    for S in seq:
        c = sum(1.0 for r in routes if set(r) <= S) / total
        level_cov.append(c)
    etas = []
    for r in routes:
        found = False
        for j, S in enumerate(seq):
            if set(r) <= S:
                etas.append(level_cov[j])
                found = True
                break
        if not found:
            etas.append(1.0)
    return etas


def main():
    g = 5
    delta = 0.10
    phi = 0.90
    Ncal = 150
    Ntest = 150
    n_outer = 250
    lam_grid = np.logspace(-4, 3, 30)
    rng_seed = np.random.default_rng(SEED)

    coverages = []
    for trial in range(n_outer):
        sd = int(rng_seed.integers(1 << 30))
        cal, nv = gen_routes(g, Ncal, delta, seed=sd)
        test, _ = gen_routes(g, Ntest, delta, seed=sd + 999)
        seq = nested_seq_from_routes(cal, nv, lam_grid)
        etas_cal = compute_eta_scores(cal, seq)
        # split-conformal quantile: ceil(phi*(N+1))-th smallest
        k = int(np.ceil(phi * (len(etas_cal) + 1))) - 1
        k = min(max(k, 0), len(etas_cal) - 1)
        etas_sorted = sorted(etas_cal)
        tau_star = etas_sorted[k]
        etas_test = compute_eta_scores(test, seq)
        cov = np.mean([1.0 if e <= tau_star + 1e-9 else 0.0 for e in etas_test])
        coverages.append(cov)
    coverages = np.array(coverages)
    mean_cov = float(coverages.mean())
    se = coverages.std(ddof=1) / np.sqrt(n_outer)
    lo = mean_cov - 1.96 * se
    phi_minus_delta = phi - delta

    # ---- Mutation: under-calibrate tau* (use 0.6*phi quantile) -> coverage should drop < phi-delta
    coverages_mut = []
    for trial in range(n_outer):
        sd = int(rng_seed.integers(1 << 30))
        cal, nv = gen_routes(g, Ncal, delta, seed=sd)
        test, _ = gen_routes(g, Ntest, delta, seed=sd + 999)
        seq = nested_seq_from_routes(cal, nv, lam_grid)
        etas_cal = compute_eta_scores(cal, seq)
        kbad = int(np.ceil(0.6 * phi * (len(etas_cal) + 1))) - 1
        kbad = min(max(kbad, 0), len(etas_cal) - 1)
        tau_bad = sorted(etas_cal)[kbad]
        etas_test = compute_eta_scores(test, seq)
        cov = np.mean([1.0 if e <= tau_bad + 1e-9 else 0.0 for e in etas_test])
        coverages_mut.append(cov)
    mean_cov_mut = float(np.mean(coverages_mut))

    verdict = "verified" if (mean_cov >= phi_minus_delta - 1e-9 and mean_cov_mut < phi_minus_delta - 1e-9) else "falsified"

    out = {
        "claim": 4,
        "statement": "Lemma 1 distribution-free marginal coverage P(B* in K_{tau*}) >= phi - delta",
        "source": "Sec.2.1 (Algorithm), proof Appendix A",
        "verdict": verdict,
        "seed": SEED,
        "setup": {"grid": g, "phi": phi, "delta": delta, "Ncal": Ncal, "Ntest": Ntest,
                  "n_outer_trials": n_outer},
        "empirical_coverage": {
            "mean": round(mean_cov, 4),
            "ci95_low": round(lo, 4),
            "std_err": round(float(se), 4),
            "phi_minus_delta": phi_minus_delta,
            "meets_bound": bool(mean_cov >= phi_minus_delta - 1e-9),
        },
        "mutation_undercalibrated": {
            "description": "Calibrate tau* with a too-small (0.6*phi) quantile instead of phi. "
                           "Coverage must drop below phi-delta.",
            "mean_coverage_undercalibrated": round(mean_cov_mut, 4),
            "phi_minus_delta": phi_minus_delta,
            "drops_below_bound": bool(mean_cov_mut < phi_minus_delta - 1e-9),
        },
    }
    with open("results/claim4.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
