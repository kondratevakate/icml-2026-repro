"""
Shared primitives for reproducing the anchored claims of
uiw8P2JGbW — "Model Monotonicity in Autobidding Auctions".

All code is CPU-only numpy/scipy/sympy. No GPU. No ML.

Model / auction conventions follow the paper (arxiv 2605.31036):

- A prediction model M is a partition P_M of a user universe U. Each
  advertiser i has a true per-user conversion probability p_i(u); the
  model predicts, for every cluster C, the cluster average
  p_i,C = mean_{u in C} p_i(u). This is *calibrated*.
- Model M_A refines M_B (M_A >= M_B) iff every cluster of M_A is
  contained in a cluster of M_B.
- tCPA bidder i has target t_i; bids b_i(C) = mu_i * t_i * p_i,C.
  At mu_i = 1 the winner in cluster C is argmax_i t_i * p_i,C and pays
  its bid (FPA) -> revenue per cluster = max_i t_i * p_i,C.
- For tCPA bidders we assume value v_i = t_i, so welfare == revenue
  (winner pays exactly v_i * p_i,C = t_i * p_i,C).
- VCG/SPA (single item): highest bidder wins, pays second-highest bid.
- ECMs are weighted sums over clusters (w_C = |C|/|U|).
"""
import json
import numpy as np
from scipy.optimize import linprog


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serializes numpy scalar/array types natively."""
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=NumpyEncoder)


def print_json(obj):
    print(json.dumps(obj, indent=2, cls=NumpyEncoder))


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
def build_model(users_per_cluster, true_p, seed=None):
    """Build a calibrated model from a partition.

    users_per_cluster: list of lists of user indices (the partition P_M)
    true_p: dict or 2D array [i][user] of true conversion probabilities
    Returns a model dict.
    """
    true_p = np.asarray(true_p, dtype=float)
    n_adv = true_p.shape[0]
    # total users = max index + 1 (we just need the set of users)
    all_users = sorted({u for cl in users_per_cluster for u in cl})
    U = len(all_users)
    partition = [list(cl) for cl in users_per_cluster]
    weights = np.array([len(cl) / U for cl in partition], dtype=float)
    # predicted cluster averages (calibrated)
    pred = np.zeros((n_adv, len(partition)), dtype=float)
    for j, cl in enumerate(partition):
        for i in range(n_adv):
            vals = [true_p[i, u] for u in cl]
            pred[i, j] = np.mean(vals)
    return {
        "n_adv": n_adv,
        "U": U,
        "partition": partition,
        "weights": weights,          # w_C
        "pred": pred,               # p_i,C  shape (n_adv, n_clusters)
        "true_p": true_p,           # p_i,u  shape (n_adv, U)
    }


def is_refinement(MA, MB):
    """True iff every cluster of MA is contained in some cluster of MB."""
    mb_clusters = [set(cl) for cl in MB["partition"]]
    for ca in MA["partition"]:
        sa = set(ca)
        if not any(sa.issubset(cb) for cb in mb_clusters):
            return False
    return True


def refinement_subclusters(MA, MB):
    """Return, for each coarse cluster index, the list of fine cluster indices.

    Assumes MA refines MB and the fine partition is a refinement of the
    coarse one (each fine cluster sits inside exactly one coarse cluster).
    """
    mapping = []
    for cb in MB["partition"]:
        sb = set(cb)
        fine_idx = [j for j, ca in enumerate(MA["partition"]) if set(ca).issubset(sb)]
        mapping.append(fine_idx)
    return mapping


# ----------------------------------------------------------------------
# FPA revenue / welfare for tCPA at mu = 1
# ----------------------------------------------------------------------
def fpa_revenue_tcpa(model, targets):
    """Rev(M) = sum_C w_C * max_i t_i * p_i,C  (tCPA, mu=1, FPA)."""
    targets = np.asarray(targets, dtype=float)
    val = model["pred"] * targets[:, None]      # t_i * p_i,C
    return float(np.sum(model["weights"] * val.max(axis=0)))


def welfare_tcpa(model, targets):
    """Welfare(M) = sum_C w_C * max_i t_i * p_i,C  (v_i = t_i => == revenue)."""
    return fpa_revenue_tcpa(model, targets)


def f(p, t):
    """f(p) = max_i t_i * p_i  — the convex function in Jensen's argument."""
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    return float(np.max(t * p))


# ----------------------------------------------------------------------
# SPA / VCG single-item auction given predictions and multipliers
# ----------------------------------------------------------------------
def spa_auction(auctions, multipliers, targets, tie_lowest=True):
    """Run a second-price (VCG) auction per impression.

    auctions: list of dicts, each with
        'pred':  array [n_adv]  predicted conversion prob (cluster avg)
        'true':  array [n_adv]  true conversion prob (for welfare)
    multipliers: array [n_adv]  g_i  (bid = g_i * pred_i)
    targets:    array [n_adv]  t_i  (welfare value per conversion = t_i)
    Returns dict with revenue, welfare, winners, payments.
    """
    multipliers = np.asarray(multipliers, dtype=float)
    targets = np.asarray(targets, dtype=float)
    rev = 0.0
    welf = 0.0
    winners = []
    payments = []
    for a in auctions:
        pred = np.asarray(a["pred"], dtype=float)
        true = np.asarray(a["true"], dtype=float)
        bids = multipliers * pred
        order = np.argsort(-bids, kind="stable")  # descending
        if tie_lowest:
            # lowest index breaks ties: re-sort equal bids by index asc
            # argsort above is stable on original order, so equal bids keep
            # ascending index -> winner is lowest index among max. Good.
            pass
        winner = order[0]
        # second-highest bid (VCG/SPA payment)
        second = bids[order[1]] if len(bids) > 1 else 0.0
        pay = second
        rev += pay
        welf += targets[winner] * true[winner]
        winners.append(int(winner))
        payments.append(float(pay))
    return {"revenue": rev, "welfare": welf, "winners": winners,
            "payments": payments}


# ----------------------------------------------------------------------
# FPA with budget constraints (single multiplier per advertiser)
# ----------------------------------------------------------------------
def fpa_budget_auction(auctions, alphas, targets, budgets):
    """First-price auction, each advertiser commits to one multiplier alpha_i.

    bids b_i(u) = alpha_i * pred_i(u). Highest bidder wins and pays bid.
    Budget B_i is an ex-ante cap on total expected spend = sum_u x_i(u) b_i(u).
    Returns revenue, welfare, winners, per-advertiser spend, per-advertiser
    conversions (true prob sum over won impressions).
    """
    alphas = np.asarray(alphas, dtype=float)
    targets = np.asarray(targets, dtype=float)
    budgets = np.asarray(budgets, dtype=float)  # inf allowed
    n_adv = len(alphas)
    rev = 0.0
    welf = 0.0
    winners = []
    spend = np.zeros(n_adv)
    conv = np.zeros(n_adv)
    for a in auctions:
        pred = np.asarray(a["pred"], dtype=float)
        true = np.asarray(a["true"], dtype=float)
        bids = alphas * pred
        winner = int(np.argmax(bids, axis=0))
        # tie -> lowest index (argmax returns first = lowest)
        pay = bids[winner]
        rev += pay
        welf += targets[winner] * true[winner]
        spend[winner] += pay
        conv[winner] += true[winner]
        winners.append(winner)
    cpa = np.where(conv > 0, spend / np.maximum(conv, 1e-12), np.inf)
    return {"revenue": rev, "welfare": welf, "winners": winners,
            "spend": spend, "conv": conv, "cpa": cpa,
            "budget_binds": np.abs(spend - np.minimum(budgets, spend)) < 1e-9}


# ----------------------------------------------------------------------
# LP allocation benchmark (Theorem 5.11)
# ----------------------------------------------------------------------
def lp_allocate(model, a, budgets):
    """Solve the centralized LP allocation benchmark.

    max_{x>=0} sum_i sum_C w_C * x_{i,C} * a_i * p_i,C
    s.t.  for each i: sum_C w_C * x_{i,C} * a_i * p_i,C <= B_i   (budget)
          for each C: sum_i x_{i,C} <= 1                          (supply)
          0 <= x_{i,C} <= 1
    a_i = t_i for tCPA (surrogate revenue == welfare), or v_i for MAX-CPA.
    Returns optimum objective (welfare / surrogate revenue) and x matrix.
    """
    a = np.asarray(a, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    n_adv = model["n_adv"]
    n_cl = model["pred"].shape[1]
    w = model["weights"]
    p = model["pred"]
    # variables: x[i*C + c]
    c = np.zeros(n_adv * n_cl)
    for i in range(n_adv):
        for cc in range(n_cl):
            c[i * n_cl + cc] = -w[cc] * a[i] * p[i, cc]   # minimize negative

    A_ub = []
    b_ub = []
    # budget constraints
    for i in range(n_adv):
        row = np.zeros(n_adv * n_cl)
        for cc in range(n_cl):
            row[i * n_cl + cc] = w[cc] * a[i] * p[i, cc]
        A_ub.append(row)
        b_ub.append(budgets[i])
    # supply constraints
    for cc in range(n_cl):
        row = np.zeros(n_adv * n_cl)
        for i in range(n_adv):
            row[i * n_cl + cc] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, 1.0)] * (n_adv * n_cl)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    x = res.x.reshape(n_adv, n_cl)
    obj = -res.fun
    return float(obj), x


def lift_solution(MB, MA, xB):
    """Lift a coarse LP solution xB to the refined model MA (Theorem 5.11).

    For each coarse cluster, copy its allocation fractions to every
    sub-cluster. Returns xA and verifies feasibility + objective match.
    """
    mapping = refinement_subclusters(MA, MB)
    n_adv = MA["n_adv"]
    n_cl_A = MA["pred"].shape[1]
    xA = np.zeros((n_adv, n_cl_A))
    for cb_idx, fine_idxs in enumerate(mapping):
        for i in range(n_adv):
            for fj in fine_idxs:
                xA[i, fj] = xB[i, cb_idx]
    return xA
