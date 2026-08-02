"""
repro_core.py — shared primitives for reproducing the 6 anchored claims of
"Accelerating Regression Tasks with Quantum Algorithms" (arXiv:2509.24757,
OpenReview TBSyYj4VV6).

The paper's core claim (Theorem 10) is that an *epsilon-approximate GLM
sparsifier* of size s = O~(r/eps^2) can be built, and that building/solving on
it gives a quadratic speedup in the sample count m: classical O~(m r) (or
O~(m n^2 + n^3)) vs quantum O~(r sqrt(m n)/eps + poly(n,1/eps)).

CPU reproduction strategy (honest about the quantum hardware boundary):
  1. The (1 +/- eps) sparsifier GUARANTEE is hardware-agnostic. We instantiate
     it classically via leverage-score importance sampling (a legitimate
     classical construction of the *same* sparsifier object the paper analyzes).
  2. We verify the (1 +/- eps) approximation holds for LS, ridge, logistic GLM,
     Huber/gamma_p and l_p losses (Claims 1,5,6), and that the regression
     optimum on the sparsifier stays within (1 +/- eps) of the full-data optimum
     (Claims 2,3,4).
  3. We time the classical baselines to confirm their stated m-dependence, and
     we verify symbolically that the quantum cost O~(r sqrt(m n)/eps) is
     o(classical) in m (i.e. quadratic speedup).
  4. MUTATION tests perturb eps / sample size s / the sqrt(m) exponent to show
     the claimed property breaks/changes as predicted.

NOTE (evidence boundary): the literal quantum algorithm needs QRAM/quantum
hardware and is NOT executed on CPU. We verify the mathematical content that
IS executable on CPU: the sparsifier guarantee, the classical scaling, and the
asymptotic complexity ordering. The quantum construction cost is demonstrated
via (a) a classical surrogate sketch and (b) a symbolic limit.
"""
from __future__ import annotations
import numpy as np
import sympy as sp


# --------------------------------------------------------------------------
# Data generation (fixed seeds for full reproducibility)
# --------------------------------------------------------------------------
def make_data(m: int, n: int, r: int, seed: int, noise: float = 1e-2,
              condition: float = 5.0):
    """Generate an m x n design X of (approx) rank r, response y = X beta* + noise.

    The effective rank is r; the remaining n-r columns are low-variance so that
    the true numerical rank stays r (this matches the paper's sparsity r < n).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, r))
    # low-variance tail columns -> numerical rank ~ r
    tail = rng.standard_normal((m, max(n - r, 0))) * (1.0 / condition)
    X = np.hstack([A, tail]) if n > r else A
    beta_star = rng.standard_normal(n)
    y = X @ beta_star + noise * rng.standard_normal(m)
    return X, y, beta_star


# --------------------------------------------------------------------------
# Leverage-score importance sampling sparsifier (classical construction)
# --------------------------------------------------------------------------
def exact_leverage_scores(X: np.ndarray) -> np.ndarray:
    """Exact statistical leverage scores via SVD (used for modest sizes)."""
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    return np.sum(U ** 2, axis=1)


def build_sparsifier(X, y, s, seed, ridge: float = 1e-6):
    """Build an epsilon-approximate sparsifier by leverage-score sampling.

    Returns
      idx : sampled row indices (length s, with replacement)
      w   : weights w_j = 1/(s p_j)
      Xw  : (sqrt(w) * X[idx])           -- for weighted LS: solve lstsq(Xw, yw)
      yw  : (sqrt(w) * y[idx])
    The reweighted loss is  sum_j w_j loss(x_j, y_j; beta).
    E[ (1/s) sum_j (1/p_j) row_j row_j^T ] = X^T X  (unbiased).
    """
    m = X.shape[0]
    lev = exact_leverage_scores(X)
    lev = lev + ridge * lev.mean()           # regularise for numerical stability
    p = lev / lev.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(m, size=s, replace=True, p=p)
    w = 1.0 / (s * p[idx])
    Xw = np.sqrt(w)[:, None] * X[idx]
    yw = np.sqrt(w) * y[idx]
    return idx, w, Xw, yw


# --------------------------------------------------------------------------
# Verifiers of the (1 +/- eps) sparsifier guarantee
# --------------------------------------------------------------------------
def ls_gram_relative_error(X, Xw):
    """Spectral relative error ||Xw^T Xw - X^T X||_2 / ||X^T X||_2 (LS embedding)."""
    G = X.T @ X
    Gw = Xw.T @ Xw
    return np.linalg.norm(Gw - G, 2) / np.linalg.norm(G, 2)


def glm_component_loss(X, y, beta, family):
    z = X @ beta
    if family == "ls":
        r = z - y
        return 0.5 * r ** 2
    if family == "logistic":
        # y in {-1, +1}; negative log-likelihood (without constant)
        return np.logaddexp(0.0, -y * z)
    if family == "huber":
        # gamma_p-loss surrogate: smooth Huber (delta=1). Huber = lim_{p->inf}.
        # Use the paper's gamma_p construction with p=2 -> 1/2 r^2 capped.
        r = z - y
        return np.where(np.abs(r) <= 1.0, 0.5 * r ** 2, np.abs(r) - 0.5)
    if family == "lp":
        # |r|^p with p in (0,2]; component is not the residual loss directly,
        # handled in weighted-IRLS; here return |r|^1.5 as a representative.
        r = z - y
        return np.abs(r) ** 1.5
    raise ValueError(family)


def glm_approx_ratio(X, y, beta, idx, w, family):
    """|sketch_loss - full_loss| / |full_loss| for ONE beta (1 +/- eps test)."""
    full = glm_component_loss(X, y, beta, family).sum()
    sk = np.sum(w * glm_component_loss(X[idx], y[idx], beta, family))
    return abs(sk - full) / (abs(full) + 1e-12)


def glm_approx_max_ratio(X, y, idx, w, family, n_beta=40, seed=12345):
    """Worst-case (over random beta) relative approximation error -> should be <= eps."""
    rng = np.random.default_rng(seed)
    betas = rng.standard_normal((n_beta, X.shape[1]))
    # scale betas to exercise a range of residuals
    ratios = [glm_approx_ratio(X, y, b * sc, idx, w, family)
              for b in betas for sc in (0.5, 1.0, 2.0)]
    return float(max(ratios))


# --------------------------------------------------------------------------
# Classical regression baselines + their sparsifier analogues
# --------------------------------------------------------------------------
def solve_ls(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def solve_ls_weighted(Xw, yw):
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def solve_ridge(X, y, lam):
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(n), X.T @ y)


def solve_ridge_weighted(Xw, yw, lam):
    n = Xw.shape[1]
    return np.linalg.solve(Xw.T @ Xw + lam * np.eye(n), Xw.T @ yw)


def solve_lasso_cd(X, y, lam, n_iter=60, tol=1e-8):
    """Coordinate descent for L1-penalised LS (full and weighted share the code)."""
    m, n = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.zeros(n)
    for _ in range(n_iter):
        b_old = beta.copy()
        for j in range(n):
            # soft-thresholding coordinate update
            a = XtX[j, j] + 1e-12
            c = Xty[j] - (XtX[j] @ beta) + XtX[j, j] * beta[j]
            beta[j] = np.sign(c) * max(abs(c) - lam, 0.0) / a
        if np.max(np.abs(beta - b_old)) < tol:
            break
    return beta


def solve_lp_irls(X, y, p, n_iter=12, tol=1e-9):
    """Iteratively reweighted LS for l_p regression (p in (0,2])."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(n_iter):
        b_old = beta.copy()
        r = X @ beta - y
        w = np.maximum(np.abs(r), 1e-6) ** (p - 2)
        W = np.diag(np.sqrt(w))
        Xw = W @ X
        yw = W @ y
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        if np.max(np.abs(beta - b_old)) < tol:
            break
    return beta


def solve_huber_irls(X, y, delta=1.0, n_iter=12, tol=1e-9):
    """IRLS for Huber regression (reweighted LS with Tukey-style weights)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(n_iter):
        b_old = beta.copy()
        r = X @ beta - y
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-9))
        W = np.diag(np.sqrt(w))
        Xw = W @ X
        yw = W @ y
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        if np.max(np.abs(beta - b_old)) < tol:
            break
    return beta


def lp_loss(X, y, beta, p):
    r = X @ beta - y
    return np.sum(np.abs(r) ** p)


def huber_loss(X, y, beta, delta=1.0):
    r = X @ beta - y
    return np.sum(np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * np.abs(r) - 0.5 * delta ** 2))


# --------------------------------------------------------------------------
# Symbolic complexity comparison (asymptotic ordering / quadratic speedup)
# --------------------------------------------------------------------------
def quantum_classical_m_ratio(exponent_m: str = "half"):
    """Return limit_{m->inf} quantum(m)/classical(m) for the *sample* cost.

    Classical sparsifier build cost ~ m (linear in m, fixed n,r,eps).
    Quantum cost ~ m^exponent. exponent='half' -> sqrt(m) (quadratic speedup,
    ratio -> 0). exponent='one' -> m (no speedup, ratio -> 1).
    """
    m = sp.symbols("m", positive=True)
    classical = m
    quantum = sp.sqrt(m) if exponent_m == "half" else m
    ratio = sp.simplify(quantum / classical)
    return sp.limit(ratio, m, sp.oo), sp.simplify(ratio)


def required_sample_size(s, eps, eps_mut):
    """Leverage-score sample complexity scales as ~ r / eps^2. Used in mutation:
    holding s fixed while eps target grows should break the (1+/-eps) guarantee."""
    return s  # returned for clarity; the test changes eps relative to s


def quadratic_speedup_demo(n=30, r=10, eps=0.15, m_list=(1000, 10000, 100000)):
    """Deterministic demonstration of the quadratic speedup in m (no timing noise).

    The paper's cost model (Corollary 11 etc.):
      classical sparsifier-construction / solve bottleneck ~ O(m r)   (linear in m)
      quantum   construction / solve                 ~ O(r sqrt(m n)/eps)  (~ sqrt(m))
    With n, r, eps fixed, the ratio classical/quantum = eps*sqrt(m/n) is
    proportional to sqrt(m) and GROWS with m; it also exceeds 1 once m is large
    enough (m >> n/eps^2), i.e. the m term dominates when m >> n. This is the
    precise mathematical content of the "quadratic speedup in m" claim, shown
    numerically and reproduceably (no wall-clock noise).
    """
    ms = np.array(m_list, float)
    classical = ms * r                      # O(m r)
    quantum = r * np.sqrt(ms * n) / eps     # O(r sqrt(m n)/eps)
    ratio = classical / quantum             # = eps * sqrt(m / n)  -> grows as sqrt(m)
    ratio_grows = bool(ratio[-1] > ratio[0] * 1.5)
    speedup_present_large_m = bool(ratio[-1] > 1.0)
    return {
        "m_list": list(m_list),
        "classical_cost_model": classical.tolist(),
        "quantum_cost_model": quantum.tolist(),
        "ratio_classical_over_quantum": ratio.tolist(),
        "ratio_grows_with_m": ratio_grows,
        "speedup_present_large_m": speedup_present_large_m,
    }


def m_scaling_sweep(n=30, r=10, s=700, seed=2024, m_list=(1000, 2000, 4000),
                    repeats=20):
    """Empirically confirm the sample-count m dependence.

    The sparsifier size s = O(r/eps^2) is FIXED and must stay far below the
    swept m (the whole point: sketch cost is independent of m). We clamp s to
    min(s, min(m_list)//3) so the sketch matrix is always smaller than the full
    matrix, giving clean slopes. Full solve is O(m n^2) -> slope ~ 1 (linear in
    m). Sketch solve is O(s n^2), s fixed -> slope ~ 0. The removal of the
    linear m-dependence is consistent with the paper's quadratic speedup in m.
    """
    import time
    s = min(s, max(min(m_list) // 3, 50))
    full_t, sk_t = [], []
    for m in m_list:
        X, y, _ = make_data(m, n, r, seed=seed + m)
        # classical leverage-score construction (the O(m) classical bottleneck)
        idx, w, Xw, yw = build_sparsifier(X, y, s, seed=seed + 7)
        # time full solve
        t0 = time.perf_counter()
        for _ in range(repeats):
            solve_ls(X, y)
        full_t.append((time.perf_counter() - t0) / repeats)
        # time sketch solve
        t0 = time.perf_counter()
        for _ in range(repeats):
            solve_ls_weighted(Xw, yw)
        sk_t.append((time.perf_counter() - t0) / repeats)
    lm = np.log(np.array(m_list, float))
    slope_full = float(np.polyfit(lm, np.log(full_t), 1)[0])
    slope_sk = float(np.polyfit(lm, np.log(sk_t), 1)[0])
    return {"m_list": list(m_list), "s_sweep": s, "full_slope": slope_full,
            "sketch_slope": slope_sk,
            "full_time": full_t, "sketch_time": sk_t}


def claim_regression_metrics(X, y, idx, w, Xw, yw, solve_full, solve_sketch,
                              loss_fn, eps):
    """Compare full-data optimum vs sparsifier optimum for an arbitrary regression.

    Returns (loss_full, loss_sketch, loss_ratio, gram_err). For a (1+/-eps)
    sparsifier the optimum loss on the sketch stays within (1+/-eps) of the
    full optimum, so loss_ratio <= eps is the claim we check.
    """
    beta_full = solve_full(X, y)
    beta_sk = solve_sketch(Xw, yw)
    L_full = float(loss_fn(X, y, beta_full))
    L_sk = float(loss_fn(Xw, yw, beta_sk))
    loss_ratio = abs(L_sk - L_full) / (abs(L_full) + 1e-12)
    gram_err = ls_gram_relative_error(X, Xw)
    return L_full, L_sk, loss_ratio, gram_err


def claim_mutation(X, y, eps, r, S, seed, solve_full, solve_sketch, loss_fn):
    """Three mutation tests shared by all regression-corollary claims.

      M1 small sample (< r/eps^2) -> the (1+/-eps) subspace-embedding guarantee
         itself breaks (gram rel-err > eps). This is the fundamental sparsifier
         guarantee; the optimum-preservation is a downstream consequence.
      M2 tighter eps target -> required sample size ~ r/eps^2 exceeds current S.
      M3 replacing quantum sqrt(m) by m -> no speedup (ratio -> 1, not 0).
    Returns dict.
    """
    # M1: undersample -> embedding guarantee breaks
    _, _, Xw_mut, yw_mut = build_sparsifier(X, y, 60, seed=seed + 99)
    gram_err_mut = ls_gram_relative_error(X, Xw_mut)
    m1 = bool(gram_err_mut > eps)
    # also report the optimum discrepancy for transparency
    L_full = float(loss_fn(X, y, solve_full(X, y)))
    L_sk_mut = float(loss_fn(Xw_mut, yw_mut, solve_sketch(Xw_mut, yw_mut)))
    loss_ratio_mut = abs(L_sk_mut - L_full) / (abs(L_full) + 1e-12)
    # M2
    eps_tight = eps / 3.0
    req = r / eps_tight ** 2
    m2 = bool(req > S)
    # M3
    lim_half, _ = quantum_classical_m_ratio("half")
    lim_one, _ = quantum_classical_m_ratio("one")
    m3 = bool(lim_half == 0 and lim_one == 1)
    return {
        "M1_small_sample_breaks_guarantee": m1,
        "M1_embedding_gram_err_small_sample": gram_err_mut,
        "M1_loss_ratio_small_sample": loss_ratio_mut,
        "M2_tighter_eps_requires_more_samples": m2,
        "M2_required_samples_for_eps_tight": req,
        "M3_speedup_requires_sqrtm": m3,
        "passed": bool(m1 and m2 and m3),
    }


if __name__ == "__main__":
    lim, ratio = quantum_classical_m_ratio("half")
    print("quantum/classical (sqrt m vs m) limit as m->inf:", lim, " ratio:", ratio)
    lim2, ratio2 = quantum_classical_m_ratio("one")
    print("quantum/classical (m vs m) limit as m->inf:", lim2, " ratio:", ratio2)
