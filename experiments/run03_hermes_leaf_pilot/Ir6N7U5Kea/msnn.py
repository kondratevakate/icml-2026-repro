"""msnn.py -- shared implementation of the MSNN / SNN estimators (Algorithms 1-3 of the paper)
and of the synthetic data generator described in Section 5.1 / Appendix B.

Everything here follows notes_paper.md; no paper text is re-read.
"""
import itertools

import numpy as np

F_SCALES = np.array([1.0, 5.0, 25.0, 625.0])       # f(d), Appendix B
P_MCAR = np.array([0.115, 0.01, 0.025, 0.05, 0.8])  # p(0)=unobserved, then low/med/high/vhigh


def make_ground_truth(m, n, r, f_scales, rng):
    """Tensor model of Assumption 2.5: A^{(d)} = U diag(lambda_d) V^T, shared row factors U.
    Entries are rescaled per level so that max |A^{(d)}| = f(d) (Assumption 4.1)."""
    U = rng.normal(size=(m, r))
    V = rng.normal(size=(n, r))
    lam = rng.uniform(0.5, 1.5, size=(len(f_scales), r))
    A = np.empty((len(f_scales), m, n))
    for d in range(len(f_scales)):
        M = (U * lam[d]) @ V.T
        A[d] = M / np.abs(M).max() * f_scales[d]
    return A, U, V, lam


def observe(A, D, sigma_rel, f_scales, rng):
    """Ytilde: observed entry at its assigned treatment level (D=0 => unobserved/nan).
    Noise: sigma_ij^{(d)} = sigma_rel * f(d)  (Assumption 4.2)."""
    m, n = D.shape
    Y = np.full((m, n), np.nan)
    for d in range(1, len(f_scales) + 1):
        mask = D == d
        if mask.any():
            noise = rng.normal(scale=sigma_rel * f_scales[d - 1], size=mask.sum())
            Y[mask] = A[d - 1][mask] + noise
    return Y


def estimate_from_anchors(S_w, q_w, x, lam_rank=None, tol=1e-9):
    """Steps 2-4 of Algorithm 2 for a single subgroup.
    S_w : |MAR| x |MAC| weighted anchor block
    q_w : |MAC|      weighted target-row vector
    x   : |MAR|      target-column vector at treatment d (unweighted)
    Returns (A_hat_k, beta_hat, rank_used)."""
    U, tau, Vt = np.linalg.svd(S_w, full_matrices=False)
    if lam_rank is None:
        lam_rank = int(np.sum(tau > tol * max(1.0, tau[0])))
    lam_rank = max(lam_rank, 1)
    # beta = pinv(S^T) q = sum_{l<=lam} (1/tau_l) u_l (v_l . q)
    beta = np.zeros(S_w.shape[0])
    for l in range(lam_rank):
        if tau[l] <= tol * max(1.0, tau[0]):
            break
        beta += (Vt[l] @ q_w) / tau[l] * U[:, l]
    return float(x @ beta), beta, lam_rank


def feasible(S_w, q_w, x, rtol=1e-6):
    """Feasibility filter of Section 5.1: x approx in Col(S), q approx in Row(S),
    and anchor matrix not of size (1,1)."""
    if S_w.shape[0] < 2 and S_w.shape[1] < 2:
        return False
    U, tau, Vt = np.linalg.svd(S_w, full_matrices=False)
    k = int(np.sum(tau > 1e-9 * max(1.0, tau[0])))
    if k == 0:
        return False
    Pcol = U[:, :k] @ U[:, :k].T
    Prow = Vt[:k].T @ Vt[:k]
    rx = np.linalg.norm(x - Pcol @ x) / max(np.linalg.norm(x), 1e-30)
    rq = np.linalg.norm(q_w - Prow @ q_w) / max(np.linalg.norm(q_w), 1e-30)
    return bool(rx < rtol and rq < rtol)


# ---------------------------------------------------------------- anchor search (Algorithm 3)
def build_B(D, i, j, d):
    """B = [ 1{D_ab = D_ib, D_aj = d, a != i, b != j} ]  (Algorithm 3, step 1)."""
    Di = D[i, :]                       # treatments of the target row
    B = (D == Di[None, :]) & (D[:, j] == d)[:, None] & (Di != 0)[None, :]
    B[i, :] = False
    B[:, j] = False
    return B


def build_B_snn(D, i, j, d):
    """Same, but restricted to a single treatment level d (Algorithm 1 / SNN)."""
    B = (D == d) & (D[:, j] == d)[:, None] & (D[i, :] == d)[None, :]
    B[i, :] = False
    B[:, j] = False
    return B


def greedy_biclique(B, max_rows=10, max_cols=10, n_restarts=8, rng=None):
    """Heuristic `maxBiclique`: the paper leaves the implementation open and notes that
    exact search is infeasible beyond size ~10.  Greedy column-growing with restarts:
    start from the column with most support, repeatedly add the column that maximises
    |rows| * |cols| of the resulting all-ones submatrix."""
    rng = rng or np.random.default_rng(0)
    cols_support = B.sum(axis=0)
    cand = np.flatnonzero(cols_support > 0)
    if cand.size == 0:
        return np.array([], int), np.array([], int)
    best = (np.array([], int), np.array([], int), 0)
    starts = cand[np.argsort(-cols_support[cand])][:n_restarts]
    for c0 in starts:
        cols = [int(c0)]
        rows = np.flatnonzero(B[:, c0])
        while len(cols) < max_cols:
            bestgain, bestc, bestrows = 0, None, None
            for c in cand:
                if c in cols:
                    continue
                nr = rows[B[rows, c]]
                if nr.size == 0:
                    continue
                score = min(nr.size, max_rows) * (len(cols) + 1)
                if score > bestgain:
                    bestgain, bestc, bestrows = score, int(c), nr
            if bestc is None or bestgain <= min(rows.size, max_rows) * len(cols):
                break
            cols.append(bestc)
            rows = bestrows
        rows_use = rows[:max_rows]
        score = rows_use.size * len(cols)
        if score > best[2]:
            best = (rows_use, np.array(sorted(cols)), score)
    return best[0], best[1]


def exhaustive_biclique(B, max_rows=6, max_cols=4):
    """Exact maximum all-ones submatrix by enumeration over column subsets (small B only)."""
    ncols = B.shape[1]
    cand = [c for c in range(ncols) if B[:, c].any()]
    best = (np.array([], int), np.array([], int), 0)
    for k in range(1, min(max_cols, len(cand)) + 1):
        for cols in itertools.combinations(cand, k):
            rows = np.flatnonzero(B[:, list(cols)].all(axis=1))
            rows = rows[:max_rows]
            score = rows.size * k
            if score > best[2]:
                best = (rows, np.array(cols), score)
    return best[0], best[1]
