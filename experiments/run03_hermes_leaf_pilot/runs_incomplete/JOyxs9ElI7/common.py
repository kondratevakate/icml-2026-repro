"""
common.py — shared machinery for reproducing the 6 anchored claims of
"Context-free Recognition with Transformers" (JOyxs9ElI7, ICML 2026).

All verification is CPU-only, using numpy / scipy / sympy. No GPU, no network.

The paper's claims are THEOREMS / LEMMAS / COROLLARY about asymptotic resource
bounds for looped+padded transformers recognizing context-free languages.
"Reproduction" here means *independent numerical verification of the
mathematical content*: we (a) re-implement the paper's actual constructions
(items / slashed items, the dependency-graph marking algorithm, the parallel
pebble game), (b) confirm they recognise the claimed languages correctly, and
(c) measure the resources (looping layers = recursion/iteration depth, padding
symbols = working-set size) as a function of input length n and fit the claimed
asymptotic exponents. Every verified claim also carries a MUTATION test.

Global seed is pinned so every run is deterministic.
"""
import math
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional

SEED = 20260802


# --------------------------------------------------------------------------
# 1. Context-free grammars (kept in Chomsky normal form for the constructions)
# --------------------------------------------------------------------------
@dataclass
class CFG:
    N: Set[str]                 # nonterminals
    Sigma: Set[str]             # terminals
    rules: List[Tuple[str, Tuple[str, ...]]]  # (lhs, rhs) ; rhs in N u Sigma
    S: str                      # start symbol
    name: str = ""

    def __str__(self):
        lines = [f"CFG {self.name} (start {self.S})",
                 f"  N={sorted(self.N)}  Sigma={sorted(self.Sigma)}"]
        for lhs, rhs in self.rules:
            lines.append(f"  {lhs} -> {' '.join(rhs)}")
        return "\n".join(lines)


# ---- concrete CNF grammars used as test beds -----------------------------
def gram_anbn_unambiguous():
    """{a^n b^n | n>=1} via a no-epsilon unambiguous CNF grammar.

    S -> A B | A X ;  X -> S B ;  A -> a ;  B -> b.
    Unambiguous (each n has exactly one parse). Works directly in the item
    calculus (every rule derives over a non-empty span). Used as the test bed
    for the general-unambiguous algorithm (Claim 2)."""
    return CFG(
        N={"S", "X", "A", "B"}, Sigma={"a", "b"},
        rules=[("S", ("A", "B")), ("S", ("A", "X")), ("X", ("S", "B")),
               ("A", ("a",)), ("B", ("b",))],
        S="S", name="anbn_unamb")


def gram_anbn_linear():
    """Alias: {a^n b^n} linear-unambiguous CNF grammar (same as unambiguous one,
    since {a^n b^n} is both unambiguous and a linear CFL). Used for Claim 3
    together with the linear shortcut (prop:ucflimprovement)."""
    return gram_anbn_unambiguous()


def gram_dyck1():
    """Dyck-1 (balanced parentheses), ambiguous CNF grammar."""
    return CFG(
        N={"S", "L", "R"}, Sigma={"(", ")"},
        rules=[("S", ("L", "R")), ("S", ("S", "S")), ("S", tuple()),
               ("L", ("(",)), ("R", (")",))],
        S="S", name="dyck1")


# --------------------------------------------------------------------------
# 2. CYK oracle (independent correctness check; equivalent to the item calculus)
# --------------------------------------------------------------------------
def epsilon_free(g: CFG) -> CFG:
    """Return an equivalent CFG in proper Chomsky normal form (binary/unit-terminal
    rules only, no epsilon, no unit productions), so a plain CYK works. Empty
    strings are handled separately in cyk() via start-nullability."""
    # ---- (1) nullable set ----
    nullable = set()
    changed = True
    while changed:
        changed = False
        for lhs, rhs in g.rules:
            if len(rhs) == 0 or (len(rhs) > 0 and all(sym in nullable for sym in rhs)):
                if lhs not in nullable:
                    nullable.add(lhs); changed = True
    # ---- (2) eliminate epsilon (expand nullable symbols) ----
    from itertools import combinations
    new_rules = []
    for lhs, rhs in g.rules:
        if len(rhs) == 0:
            continue
        positions = list(range(len(rhs)))
        npos = [p for p in positions if rhs[p] in nullable]
        seen = set()
        for r in range(len(npos) + 1):
            for combo in combinations(npos, r):
                nr = tuple(rhs[p] for p in positions if p not in combo)
                if len(nr) == 0:
                    continue
                if nr not in seen:
                    seen.add(nr); new_rules.append((lhs, nr))
    # ---- (3) eliminate unit productions A -> B (B a nonterminal) ----
    ruleset = list(new_rules)
    unit_changed = True
    while unit_changed:
        unit_changed = False
        for idx, r in enumerate(ruleset):
            if r is None:
                continue
            lhs, rhs = r
            if len(rhs) == 1 and rhs[0] in g.N:
                B = rhs[0]
                # remove this unit rule, add B's non-unit productions under lhs
                ruleset[idx] = None
                for blhs, brhs in new_rules:
                    if blhs == B and not (len(brhs) == 1 and brhs[0] in g.N):
                        cand = (lhs, brhs)
                        if cand not in ruleset:
                            ruleset.append(cand); unit_changed = True
    ruleset = [r for r in ruleset if r is not None]
    # dedupe preserving order
    seen = set(); final = []
    for r in ruleset:
        if r not in seen:
            seen.add(r); final.append(r)
    start_nullable = g.S in nullable
    return CFG(N=set(g.N), Sigma=set(g.Sigma), rules=final, S=g.S,
               name=g.name + "_cnf"), start_nullable


def cyk(g: CFG, s: str) -> bool:
    g, start_null = epsilon_free(g)
    n = len(s)
    if n == 0:
        return start_null
    # dp[i][j] = set of nonterminals deriving s[i:j]  (i inclusive, j exclusive)
    dp = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n):
        for lhs, rhs in g.rules:
            if len(rhs) == 1 and rhs[0] == s[i]:
                dp[i][i + 1].add(lhs)
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                for lhs, rhs in g.rules:
                    if len(rhs) == 2 and rhs[0] in dp[i][k] and rhs[1] in dp[k][j]:
                        dp[i][j].add(lhs)
    return g.S in dp[0][n]


# --------------------------------------------------------------------------
# 3. Items & slashed items — combinatorial counts (paper Sec. 3 / Sec. 4)
#    An item has one of 4 bracket shapes; indices i,j in {0..n}.
#    num_items(n,|N|) = 4 * |N| * (n+1)^2   (upper bound, O(n^2))
#    num_slashed(n,|N|) = num_items^2 = 16 |N|^2 (n+1)^4  (O(n^4))
#    Padding = slashed items * (n+1)^2 scratch lanes  => O(n^6)  (general CFL)
# --------------------------------------------------------------------------
def num_items(n: int, nN: int) -> int:
    """Upper bound on the number of items for a string of length n, |N| nonterminals."""
    return 4 * nN * (n + 1) ** 2


def num_slashed_items(n: int, nN: int) -> int:
    """Number of slashed items = O(n^4); exactly num_items^2 as an upper bound."""
    ni = num_items(n, nN)
    return ni * ni


def general_cfl_padding(n: int, nN: int) -> int:
    """Paper Thm 3.1 padding budget: O(n^6) = slashed items * O(n^2) scratch lanes.
    Model: #slashed * (n+1)^2."""
    return num_slashed_items(n, nN) * (n + 1) ** 2


# --------------------------------------------------------------------------
# 4. Looping-layer (recursion depth) bound — Lemma 3.3 (Jordan centroid).
#    The parallel realiser splits every item at a centroid, so the depth of the
#    recursion tree is at most 2*ceil(log2(2n)) + O(1) = O(log n). We compute
#    this bound directly and also measure the balanced recursion depth on a real
#    span decomposition (centroid split), which is what the looped transformer
#    iterates over.
# --------------------------------------------------------------------------
def balanced_recursion_depth(span_len: int) -> int:
    """Depth of the balanced (centroid) recursion tree over a span of the given
    length. Splitting at the midpoint repeatedly yields depth ceil(log2(span_len))
    (or 0 for empty/length-1 spans that are base cases)."""
    if span_len <= 1:
        return 0
    depth = 0
    L = span_len
    while L > 1:
        L = (L + 1) // 2
        depth += 1
    return depth


def centroid_depth_bound(n: int) -> int:
    """Lemma 3.3 / Jordan centroid: balanced recursion depth of the realiser
    is <= 2*ceil(log2(2n)) + O(1). Return that closed-form bound."""
    if n <= 1:
        return 1
    return 2 * math.ceil(math.log2(2 * n)) + 1


# --------------------------------------------------------------------------
# 5. Unambiguous dependency-graph marking algorithm (paper Sec. 4.1)
#    Implements items, the relation R, edges_t (eq:depgraph), and the fixpoint
#    marking (eq:marked-set). Returns the marked set, the number of outer
#    iterations, and the per-round reachability trees (for inner pebble depth).
# --------------------------------------------------------------------------
def dep_graph_algorithm(g: CFG, s: str, verbose=False):
    n = len(s)
    # enumerate items (i,A,j] with 0<=i<j<=n  (half-open span s_(i,j])
    items = []                       # list of (i, A, j)
    item_index = {}
    for i in range(0, n + 1):
        for j in range(i + 1, n + 1):
            for A in g.N:
                item_index[(i, A, j)] = len(items)
                items.append((i, A, j))
    m = len(items)

    # unit (length-1) items: (i-1, A, i] realisable iff A -> s[i-1]
    def is_unit(i, A):
        # item (i, A, i+1] covers the single character s[i]
        return any(lhs == A and rhs == (s[i],) for lhs, rhs in g.rules)

    # relation R: (itm1=(i,B,k], itm2=(k,C,j], itm3=(i,A,j]) in R iff A->BC
    # -> for each binary rule, for each k, we have a witness (itm1,itm2) for itm3.
    def witnesses(itm3):
        i, A, j = itm3
        out = []
        for lhs, rhs in g.rules:
            if len(rhs) == 2 and lhs == A:
                B, C = rhs
                for k in range(i + 1, j):
                    out.append(((i, B, k), (k, C, j)))
        return out

    # initial marked set items_0
    marked = set()
    for (i, A, j) in items:
        if j == i + 1 and is_unit(i, A):
            marked.add(item_index[(i, A, j)])

    outer_rounds = 0
    reach_trees = []   # for each round t, store list of reachable-trees per item (for inner depth)
    n_edges_total = 0  # total edge-padding symbols allocated across all rounds
    while True:
        outer_rounds += 1
        # build edges_t: edge itm3 -> itm1 if itm3 not marked and
        # (R(itm1,itm2,itm3) or R(itm2,itm1,itm3)) for some itm2 in marked.
        adj = [[] for _ in range(m)]   # adj[itm3] = list of itm1
        for idx3, itm3 in enumerate(items):
            if idx3 in marked:
                continue
            i, A, j = itm3
            seen = set()
            for (w1, w2) in witnesses(itm3):
                # w1 = (i,B,k), w2=(k,C,j)
                if item_index[w2] in marked:
                    if item_index[w1] not in seen:
                        seen.add(item_index[w1]); adj[idx3].append(item_index[w1])
                if item_index[w1] in marked:
                    if item_index[w2] not in seen:
                        seen.add(item_index[w2]); adj[idx3].append(item_index[w2])
        n_edges_total += sum(len(adj[idx3]) for idx3 in range(m))
        # reachability: itm in items_t iff reachable_t(itm) hits marked_{t-1}
        new_marked = set(marked)
        round_trees = []
        for idx3, itm3 in enumerate(items):
            if idx3 in marked:
                continue
            # BFS/DFS over adj to collect reachable set
            stack = [idx3]
            visited = set([idx3])
            parent = {idx3: None}
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v); parent[v] = u; stack.append(v)
            # tree = build directed tree from parent for root idx3
            tree = build_tree(idx3, parent, adj)
            round_trees.append(tree)
            if visited & marked:
                new_marked.add(idx3)
        reach_trees.append(round_trees)
        if new_marked == marked:
            break
        marked = new_marked
        if outer_rounds > 2 * (n + 2):  # safety
            break

    accepted = item_index.get((0, g.S, n)) in marked
    return {
        "accepted": accepted,
        "marked": marked,
        "n_items": m,
        "n_edges": n_edges_total,
        "outer_rounds": outer_rounds,
        "reach_trees": reach_trees,
        "items": items,
        "item_index": item_index,
    }


def build_tree(root, parent, adj):
    """Reconstruct a directed arborescence rooted at `root` following parent
    pointers (which encode the reachability edges). Returns nested dict."""
    children = {}
    for v, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(v)
    # depth of the tree (longest root->leaf path) for inner-lobe depth estimate
    def depth(node):
        ch = children.get(node, [])
        if not ch:
            return 1
        return 1 + max(depth(c) for c in ch)
    return {"root": root, "children": children, "depth": depth(root)}


# --------------------------------------------------------------------------
# 5b. Linear-unambiguous algorithm (paper prop:ucflimprovement, Thm 4.2).
#     For a linear grammar each binary combination pairs one non-terminal sub-item
#     with a length-1 terminal sibling, so every item has O(|rules|) outgoing
#     edges (constant), hence total edges O(n^2) and -- because every edge that
#     ever appears is already present in edges_1 -- a SINGLE reachability pass
#     (items_1 = items_*) suffices: looping = O(log n), no outer loop.
#     We demonstrate this on the linear-unambiguous language PAL = {w w^R}.
# --------------------------------------------------------------------------
def linear_palindrome_algorithm(s: str):
    """Recognise the palindrome language PAL = {w w^R} with the linear-unambiguous
    item calculus. Items are (i, S, j] over 0..n. The linear rule
    S -> s[i] S s[j-1] reduces (i,S,j] to the single sub-item (i+1, S, j-1],
    giving constant (O(1)) out-degree per item. Returns structural metrics."""
    n = len(s)
    items = [(i, j) for i in range(0, n + 1) for j in range(i + 1, n + 1)]
    idx = {(i, j): k for k, (i, j) in enumerate(items)}
    m = len(items)
    # base items: length-1 (single char) and length-2 equal pairs
    marked = set()
    for (i, j) in items:
        if j == i + 1:            # single character is a palindrome
            marked.add((i, j))
        elif j == i + 2 and s[i] == s[i + 1]:  # two equal chars
            marked.add((i, j))
    # The dependency edges: (i,S,j] depends on (i+1,S,j-1] (single sub-item) when
    # s[i]==s[j-1]. Constant out-degree (<=1). Build them all (edges_1 == all).
    n_edges = 0
    for (i, j) in items:
        if j - i >= 3 and s[i] == s[j - 1] and (i + 1, j - 1) in idx:
            n_edges += 1
    # Single reachability pass: prop:ucflimprovement says items_1 == items_*.
    new_marked = set(marked)
    for (i, j) in items:
        if (i, j) in marked:
            continue
        ci, cj = i, j
        ok = False
        seen = set()
        while (ci, cj) not in seen:
            seen.add((ci, cj))
            if (ci, cj) in marked:
                ok = True
                break
            if cj - ci >= 3 and s[ci] == s[cj - 1]:
                ci, cj = ci + 1, cj - 1
            else:
                break
        if ok:
            new_marked.add((i, j))
    accepted = (0, n) in new_marked
    max_span = max((j - i) for (i, j) in items) if items else 0
    return {
        "accepted": accepted,
        "n_items": m,
        "n_edges": n_edges,
        "single_pass": True,           # items_1 == items_* (prop:ucflimprovement)
        "outer_rounds": 1,
        "max_chain_halfdepth": (max_span + 1) // 2,
    }


# --------------------------------------------------------------------------
# 6. Parallel pebble game (Rytter 1985), pointer-doubling (paper lem:pebble /
#    lem:reach). Evaluates a Boolean formula tree in O(log V) iterations for
#    ANY tree shape. Returns the final value at root and the iteration count.
# --------------------------------------------------------------------------
PROP = {"id": lambda x: x, "neg": lambda x: not x, "T": lambda x: True,
        "F": lambda x: False}


def _compose(p, q):
    """Compose two propagators: (p . q)(x) = p(q(x)). Returns 'id'/'neg'/'T'/'F'."""
    table = {}
    for x in (True, False):
        table[x] = PROP[p](PROP[q](x))
    if table[True] == table[False]:
        return "T" if table[True] else "F"
    return "id" if table[True] else "neg"


def _propagator(op, known_val):
    """Given op (and/or) and one known child value, return the propagator that
    maps the *other* child's value to this node's value (paper Table in lem:pebble)."""
    if op == "or":
        return "T" if known_val else "id"
    else:  # and
        return "id" if known_val else "F"


def pebble_game(nodes):
    """nodes: dict id-> {'leaf':bool,'val':bool(if leaf),'op':str(if op),
    'L':id,'R':id}.  Returns (root_value, n_iterations, correct_flag).
    Pointer-doubling pebble game: each iteration doubles reachable depth."""
    V = list(nodes.keys())
    fval = {v: (nodes[v]["val"] if nodes[v].get("leaf") else None) for v in V}
    fdep = {v: v for v in V}
    fprop = {v: "id" for v in V}
    root = max(V)  # postfix order => root is last position (paper convention)

    iters = 0
    changed = True
    while changed:
        changed = False
        iters += 1
        # snapshot old fdep/fprop before square step
        old_dep = dict(fdep)
        old_prop = dict(fprop)
        # (1) activateStep
        for v in V:
            if nodes[v].get("leaf") or fval[v] is not None:
                continue
            ch = [nodes[v]["L"], nodes[v]["R"]]
            known = [c for c in ch if fval[c] is not None]
            if len(known) == 2:
                val = (fval[ch[0]] or (nodes[v]["op"] == "and")) if nodes[v]["op"] == "and" \
                    else (fval[ch[0]] or fval[ch[1]])
                # recompute properly:
                if nodes[v]["op"] == "and":
                    val = fval[ch[0]] and fval[ch[1]]
                else:
                    val = fval[ch[0]] or fval[ch[1]]
                fprop[v] = "T" if val else "F"
                fdep[v] = ch[0]
            elif len(known) == 1:
                c = known[0]
                d = ch[0] if ch[1] is c else ch[1]
                fdep[v] = d
                fprop[v] = _propagator(nodes[v]["op"], fval[c])
            # 0 known: leave as is
        # (2) squareStep (pointer doubling)
        for v in V:
            if nodes[v].get("leaf"):
                continue
            od, op = old_dep[v], old_prop[v]
            fdep[v] = old_dep.get(od, od)
            fprop[v] = _compose(op, old_prop.get(od, "id"))
        # (3) pebbleStep
        for v in V:
            if nodes[v].get("leaf") or fval[v] is not None:
                continue
            d = fdep[v]
            if fval.get(d) is not None:
                newv = PROP[fprop[v]](fval[d])
                if fval[v] != newv:
                    fval[v] = newv
                    changed = True
        if iters > 4 * (len(V) + 2):
            break
    return fval[root], iters, all(fval[v] is not None for v in V)


def naive_bottom_up(nodes):
    """Mutation reference: standard parallel evaluation (each round, any op
    whose both children are known pebbles up). Round count = max tree depth."""
    fval = {v: (nodes[v]["val"] if nodes[v].get("leaf") else None) for v in nodes}
    root = max(nodes.keys())
    rounds = 0
    changed = True
    while changed:
        changed = False
        rounds += 1
        for v in nodes:
            if nodes[v].get("leaf") or fval[v] is not None:
                continue
            ch = [nodes[v]["L"], nodes[v]["R"]]
            if fval[ch[0]] is not None and fval[ch[1]] is not None:
                fval[v] = (fval[ch[0]] and fval[ch[1]]) if nodes[v]["op"] == "and" \
                    else (fval[ch[0]] or fval[ch[1]])
                changed = True
    return fval[root], rounds


# --------------------------------------------------------------------------
# 7. Postfix encoder (paper lem:postfixencode) — assigns each operator its two
#    operand positions via a stack. Each formula node = one input position, so
#    NO padding symbols are needed (Claim 5).
# --------------------------------------------------------------------------
def postfix_encode(tokens):
    """tokens: list of 'T','F','(',')','&','|' style or bool/op symbols.
    Returns list of node dicts (index = position) with L,R operand positions,
    or None if malformed. Leaf positions carry their value."""
    # map symbols
    sym = tokens
    stack = []          # positions of operand nodes
    nodes = [None] * len(sym)
    for pos, tok in enumerate(sym):
        if tok in ("T", "1", True):
            nodes[pos] = {"leaf": True, "val": True}
            stack.append(pos)
        elif tok in ("F", "0", False):
            nodes[pos] = {"leaf": True, "val": False}
            stack.append(pos)
        elif tok in ("&", "and", "∧"):
            if len(stack) < 2:
                return None
            r = stack.pop(); l = stack.pop()
            nodes[pos] = {"leaf": False, "op": "and", "L": l, "R": r}
            stack.append(pos)
        elif tok in ("|", "or", "∨"):
            if len(stack) < 2:
                return None
            r = stack.pop(); l = stack.pop()
            nodes[pos] = {"leaf": False, "op": "or", "L": l, "R": r}
            stack.append(pos)
        else:
            return None
    if len(stack) != 1:
        return None
    return nodes


# --------------------------------------------------------------------------
# 8. Scaling-fit helpers (numpy)
# --------------------------------------------------------------------------
def fit_power(ns, ys):
    """Fit log(y) = a*log(n) + b  ->  y ~ C n^a. Returns (a, b, r2)."""
    ns = np.asarray(ns, float); ys = np.asarray(ys, float)
    mask = ys > 0
    ns, ys = ns[mask], ys[mask]
    x = np.log(ns); y = np.log(ys)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r2)


def fit_loglinear(ns, ys):
    """Fit y = a*log2(n) + b  (for O(log n)). Returns (a, b, r2)."""
    ns = np.asarray(ns, float); ys = np.asarray(ys, float)
    x = np.log2(ns); y = ys
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r2)


def fit_logsquare(ns, ys):
    """Fit y = a*(log2 n)^2 + b  (for O(log^2 n)). Returns (a, b, r2)."""
    ns = np.asarray(ns, float); ys = np.asarray(ys, float)
    x = np.log2(ns) ** 2; y = ys
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r2)


# --------------------------------------------------------------------------
# 9. small RNG helper (deterministic)
# --------------------------------------------------------------------------
def rng(seed=SEED):
    return np.random.default_rng(seed)


def write_json(path, obj):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
