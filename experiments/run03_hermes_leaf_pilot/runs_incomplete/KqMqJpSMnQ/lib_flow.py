"""Max-flow / min-cut (Dinic, pure python) + parametric min-cut network of paper Sec.4.2.

Network D_lambda (Gallo-Grigoriadis-Tarjan style) for hypergraph H=(V,E):
  nodes: s, t, u_e (one per hyperedge e), n_v (one per vertex v)
  edges:
    (s, u_e)   capacity  lambda * w_e
    (n_v, t)   capacity  1
    (u_e, n_v) capacity  INF   for each v in e
Min s-t cut capacity = |K| + lambda*(W - e(K)) where K = {v : n_v on source side}.
So minimizing the cut over K is exactly the Lagrangian L(lambda) of the LP coverage constraint.
As lambda increases, the optimal K grows => nested family of subgraphs (Lemma 2 / Theorem 2).
"""
from collections import deque


class Dinic:
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.graph = [[] for _ in range(n_nodes)]  # adjacency: list of edge indices
        self.edges = []                            # each edge: [to, cap]

    def add_edge(self, u, v, cap):
        self.graph[u].append(len(self.edges))
        self.edges.append([v, float(cap)])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0.0])

    def _bfs(self, s, t, level):
        for i in range(self.n):
            level[i] = -1
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for ei in self.graph[u]:
                v, cap = self.edges[ei]
                if cap > 1e-12 and level[v] == -1:
                    level[v] = level[u] + 1
                    q.append(v)
        return level[t] != -1

    def _dfs(self, u, t, pushed, level, it):
        if u == t:
            return pushed
        while it[u] < len(self.graph[u]):
            ei = self.graph[u][it[u]]
            v, cap = self.edges[ei]
            if cap > 1e-12 and level[v] == level[u] + 1:
                d = self._dfs(v, t, min(pushed, cap), level, it)
                if d > 1e-12:
                    self.edges[ei][1] -= d
                    self.edges[ei ^ 1][1] += d
                    return d
            it[u] += 1
        return 0.0

    def max_flow(self, s, t):
        flow = 0.0
        level = [-1] * self.n
        INF = float('inf')
        while self._bfs(s, t, level):
            it = [0] * self.n
            while True:
                pushed = self._dfs(s, t, INF, level, it)
                if pushed <= 1e-12:
                    break
                flow += pushed
        return flow

    def source_side(self, s):
        """Vertices reachable from s in residual graph (the source side of the min cut)."""
        seen = [False] * self.n
        q = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            for ei in self.graph[u]:
                v, cap = self.edges[ei]
                if cap > 1e-12 and not seen[v]:
                    seen[v] = True
                    q.append(v)
        return seen


def build_flow_network(vertices, hyperedges, weights, lam):
    """Build D_lambda. Returns (Dinic, n_vertices list-of-node-indices, node-count, S, T)."""
    n = len(vertices)
    m = len(hyperedges)
    S = 0
    T = 1
    u0 = 2
    n0 = 2 + m
    N = 2 + m + n
    INF = 1e18
    d = Dinic(N)
    for i, e in enumerate(hyperedges):
        d.add_edge(S, u0 + i, lam * weights[i])
    for v in range(n):
        d.add_edge(n0 + v, T, 1.0)
    for i, e in enumerate(hyperedges):
        for v in e:
            d.add_edge(u0 + i, n0 + v, INF)
    n_node = [n0 + v for v in range(n)]
    return d, n_node, N, S, T


def min_cut_K(vertices, hyperedges, weights, lam):
    """Return the vertex set K = {v : n_v on source side} minimizing the Lagrangian at lambda."""
    d, n_node, N, S, T = build_flow_network(vertices, hyperedges, weights, lam)
    d.max_flow(S, T)
    side = d.source_side(S)
    K = set(vertices[v] for v in range(len(vertices)) if side[n_node[v]])
    return K


def parametric_sequence(vertices, hyperedges, weights, lam_grid):
    """Sweep lambda in increasing order. Returns the ordered nested list of distinct vertex sets S_0..S_k."""
    seq = []
    prev = None
    for lam in lam_grid:
        K = min_cut_K(vertices, hyperedges, weights, lam)
        if prev is None or K != prev:
            seq.append(K)
            prev = K
    return seq


if __name__ == "__main__":
    vs = [0, 1, 2]
    hes = [(0, 1), (1, 2)]
    ws = [1.0, 1.0]
    for lam in [0.1, 1.0, 10.0]:
        print(lam, sorted(min_cut_K(vs, hes, ws, lam)))
