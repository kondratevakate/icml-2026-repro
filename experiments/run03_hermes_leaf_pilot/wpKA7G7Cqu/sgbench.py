"""sgbench.py -- CPU-only reproduction substrate for SGShift (wpKA7G7Cqu).

Implements, from first principles (numpy/scipy only):

  * Semi-synthetic concept-shift benchmark with KNOWN shifted-feature ground truth
    (mirrors the paper's "generator" + "base model" construction, Sec. 4 / Appendix G).
  * SGShift (sparse GAM correction, Eq. 2), SGShift-A (absorption, Eq. 3-4),
    SGShift-K (Model-X knockoffs, Eq. 5-6), and SGShift-KA (knockoffs + absorption).
  * Baselines: Diff, WhyShift (CART on model-difference), SHAP-difference.
  * Penalized GLM solver via FISTA (logistic + linear).
  * Model-X second-order Gaussian knockoffs (conditional sampler).
  * Metrics: AUROC (rank-based) and recall at fixed FPR=5%.

All randomness is seeded. No GPU, no sklearn -- only numpy/scipy/sympy.
"""
import numpy as np
from numpy.linalg import LinAlgError

rng_global = np.random.default_rng(20260523)


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def soft(x, thr):
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def make_sigma(p, rho):
    """AR(1) covariance: Sigma_{ij} = rho^{|i-j|} (plus tiny ridge for PSD)."""
    idx = np.arange(p)
    S = rho ** np.abs(idx[:, None] - idx[None, :])
    S = S + 1e-6 * np.eye(p)
    return S


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def standardize(X, mu, sd):
    return (X - mu) / sd


def auc_score(labels, scores):
    """Rank-based AUROC (Mann-Whitney). labels: 1=positive, 0=negative."""
    labels = np.asarray(labels, float)
    scores = np.asarray(scores, float)
    pos = labels == 1
    neg = ~pos
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average rank (ties get mean rank)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    s_sorted = scores[order]
    # assign average ranks for ties
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def recall_at_fpr(labels, scores, fpr=0.05):
    """Recall at a FIXED false-positive rate `fpr`: greedily select features from
    the top of the ranked list, stopping as soon as adding one more NULL feature
    would push the selected-set FPR above `fpr`. This recovers the largest
    high-ranking set with FPR <= fpr (matches the paper's recall@FPR5 protocol),
    so with a perfect ranking all true features are recovered (recall -> 1)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n_null = int((labels == 0).sum())
    n_true = int((labels == 1).sum())
    if n_true == 0:
        return float("nan")
    max_null = fpr * n_null
    order = np.argsort(-scores, kind="mergesort")  # descending
    sel_null = 0
    sel_true = 0
    for idx in order:
        if labels[idx] == 0:
            if sel_null + 1 > max_null + 1e-9:
                break
            sel_null += 1
        else:
            sel_true += 1
    return float(sel_true / n_true)


# --------------------------------------------------------------------------- #
# penalized GLM solver (FISTA)
# --------------------------------------------------------------------------- #
def _lipschitz(Z, family, n):
    if family == "linear":
        e = np.linalg.eigvalsh(Z.T @ Z / n)
        return float(e.max()) + 1e-8
    else:
        s = np.linalg.svd(Z, compute_uv=False).max()
        return float(s * s) / (4.0 * n) + 1e-8


def fit_penalized(Z, y, offset, lam, family="logistic", lam2=None,
                  max_iter=500, tol=1e-5, beta0=None):
    """FISTA for  min_beta  1/n sum_i loss(offset + Z beta, y) + lam*||beta||_1.
    `lam2` (optional, same shape as beta) gives per-coordinate penalties for
    the two-block (absorption) design.
    Returns beta (array)."""
    n, p = Z.shape
    Zt = Z.T
    if beta0 is None:
        beta = np.zeros(p)
    else:
        beta = beta0.copy()
    if lam2 is None:
        pen = lam
    else:
        pen = np.broadcast_to(lam2, p).astype(float)
    L = _lipschitz(Z, family, n)
    step = 1.0 / L
    z = beta.copy()
    t = 1.0
    prev = beta.copy()
    for _ in range(max_iter):
        eta = offset + Z @ z
        if family == "logistic":
            s = 1.0 / (1.0 + np.exp(-eta))
            grad = Zt @ (s - y) / n
        else:
            grad = Zt @ (eta - y) / n
        if lam2 is None:
            bnew = soft(z - step * grad, step * lam)
        else:
            bnew = np.sign(z - step * grad) * np.maximum(np.abs(z - step * grad) - step * pen, 0.0)
        tnew = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = bnew + ((t - 1.0) / tnew) * (bnew - beta)
        beta, t = bnew, tnew
        if np.max(np.abs(beta - prev)) < tol:
            break
        prev = beta.copy()
    return beta


def _lam_max(Z, y, offset, family):
    """Smallest lambda that forces all-zero solution (null gradient magnitude)."""
    n = Z.shape[0]
    eta = offset
    if family == "logistic":
        s = 1.0 / (1.0 + np.exp(-eta))
        g = Z.T @ (s - y) / n
    else:
        g = Z.T @ (eta - y) / n
    return float(np.max(np.abs(g)))


def logistic_loss(y, eta):
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta))


# --------------------------------------------------------------------------- #
# Model-X second-order Gaussian knockoffs (conditional sampler)
# --------------------------------------------------------------------------- #
def gaussian_knockoffs(X, rng):
    """Sample knockoff copy Xtil of X (n x p) under the Gaussian Model-X
    assumption. Uses the equicorrelated construction:
        Xtil | X ~ N( (Sigma - S) Sigma^{-1} X ,  R ),  S = s I, s = lambda_min(Sigma).
    Satisfies the knockoff exchangeability property (Cov(X_j, Xtil_k)=Cov(X_j,X_k) for j!=k
    and Cov(X_j,Xtil_j)=Sigma_jj - s)."""
    n, p = X.shape
    Sigma = np.cov(X, rowvar=False) + 1e-6 * np.eye(p)
    try:
        chol = np.linalg.cholesky(Sigma)
    except LinAlgError:
        Sigma = Sigma + 1e-3 * np.eye(p)
        chol = np.linalg.cholesky(Sigma)
    # equicorrelated: s = lambda_min(Sigma)
    w, V = np.linalg.eigh(Sigma)
    s = max(w.min(), 1e-3)
    S = s * np.eye(p)
    G = Sigma - S
    # R = 2S - S Sigma^{-1} S  (Schur complement); build via eigendecomp
    Sinv = (V / w) @ V.T
    R = 2.0 * S - S @ Sinv @ S
    R = (R + R.T) / 2.0
    wR, VR = np.linalg.eigh(R)
    wR = np.clip(wR, 1e-8, None)
    L22 = VR * np.sqrt(wR)
    Z = rng.standard_normal((n, p))
    Xtil = (G @ Sinv) @ X.T           # (Sigma - S) Sigma^{-1} X
    Xtil = Xtil.T + Z @ L22.T
    return Xtil


# --------------------------------------------------------------------------- #
# dataset generator (semi-synthetic concept shift, known ground truth)
# --------------------------------------------------------------------------- #
def gen_dataset(p=30, n_S=2000, n_T=2000, rho=0.3,
                source_active=None, shift_set=None, shift_mag=1.0,
                mismatch=False, match_seed=0, seed=0, family="logistic"):
    """Generate a source/target concept-shift dataset with KNOWN shifted set.

    - True source:  eta_S(x) = sum_{j in source_active} c_j x_j   (linear GAM)
      If mismatch: h_S is trained with heavy L1 regularization so it UNDERFITS the
      source, dropping some source-active features; that dropped (shared, linear)
      component becomes source-model misfit that SGShift-A's absorption term should
      absorb. This is a clean, fully-linear way to realize "generator != base model".
    - Concept shift:  eta_T(x) = eta_S(x) + sum_{j in shift_set} d_j x_j.
    - Marginal P(X) identical in S and T (pure concept shift, as defined in Sec. 2).
    - Standardization applied within the pipeline (paper standardizes features).
    """
    rng = np.random.default_rng([seed, match_seed, p, n_S, n_T, int(rho * 100),
                                 int(mismatch)])
    Sigma = make_sigma(p, rho)
    chol = np.linalg.cholesky(Sigma)
    X_S = (rng.standard_normal((n_S, p)) @ chol.T)
    X_T = (rng.standard_normal((n_T, p)) @ chol.T)

    if source_active is None:
        source_active = list(range(10))
    if shift_set is None:
        shift_set = list(range(20, 25))
    shift_set = list(shift_set)

    # source coefficients
    c = np.zeros(p)
    for j in source_active:
        c[j] = rng.uniform(0.7, 1.4) * (1 if rng.random() < 0.5 else -1)

    # shift coefficients (the ground-truth concept shift)
    d = np.zeros(p)
    for j in shift_set:
        d[j] = (shift_mag if rng.random() < 0.5 else -shift_mag)

    eta_S = X_S @ c
    eta_T = X_T @ c + X_T @ d

    if family == "logistic":
        Y_S = (rng.random(n_S) < 1.0 / (1.0 + np.exp(-eta_S))).astype(float)
        Y_T = (rng.random(n_T) < 1.0 / (1.0 + np.exp(-eta_T))).astype(float)
    else:  # linear regression (SUPPORT2-style)
        Y_S = eta_S + rng.normal(0.0, 0.5, n_S)
        Y_T = eta_T + rng.normal(0.5, 0.0, n_T)

    # standardize using SOURCE statistics (as in paper pipeline)
    mu, sd = standardize_fit(X_S)
    Xs_S = standardize(X_S, mu, sd)
    Xs_T = standardize(X_T, mu, sd)

    # ---- train source model h_S on source data ----
    lam_hs = 1e-3 if not mismatch else 0.15   # heavy reg -> underfit in mismatch
    hS_beta = fit_penalized(Xs_S, Y_S, np.zeros(n_S), lam_hs, family=family, max_iter=1500)
    hS_S = Xs_S @ hS_beta
    hS_T = Xs_T @ hS_beta

    # training misfit (shared across domains by construction)
    if family == "logistic":
        train_err = logistic_loss(Y_S, hS_S)
    else:
        train_err = 0.5 * np.mean((Y_S - hS_S) ** 2)

    return dict(
        p=p, Xs_S=Xs_S, Y_S=Y_S, Xs_T=Xs_T, Y_T=Y_T,
        hS_S=hS_S, hS_T=hS_T, hS_beta=hS_beta,
        shift_set=shift_set, true_delta=d, source_active=source_active,
        Sigma=Sigma, train_err=float(train_err), mismatch=mismatch,
        family=family, lam_hs=lam_hs,
    )


# --------------------------------------------------------------------------- #
# method implementations
# --------------------------------------------------------------------------- #
def _corr_lambda(Z, y, offset, family, frac=0.1):
    return frac * _lam_max(Z, y, offset, family)


def method_sg(D, lam=None, frac=0.08, family=None):
    """SGShift (Eq. 2): penalized correction on TARGET only, offset = h_S."""
    fam = family or D["family"]
    Z = D["Xs_T"]
    offset = D["hS_T"]
    y = D["Y_T"]
    if lam is None:
        lam = _corr_lambda(Z, y, offset, fam, frac)
    beta = fit_penalized(Z, y, offset, lam, family=fam)
    return np.abs(beta), dict(lam=lam)


def method_sga(D, lam_d=None, frac=0.08, c=0.3, family=None):
    """SGShift-A (Eq. 3-4): absorption. Shared omega (both domains) + target delta."""
    fam = family or D["family"]
    ZS, ZT = D["Xs_S"], D["Xs_T"]
    yS, yT = D["Y_S"], D["Y_T"]
    offS, offT = D["hS_S"], D["hS_T"]
    nS, nT = ZS.shape[0], ZT.shape[0]
    p = ZS.shape[1]
    # augmented design: [ZS, 0] over source ; [ZT, ZT] over target
    Z_aug = np.vstack([np.hstack([ZS, np.zeros_like(ZS)]),
                       np.hstack([ZT, ZT])])
    off_aug = np.concatenate([offS, offT])
    y_aug = np.concatenate([yS, yT])
    # weights w_S = w_T = 1/2 -> handled by averaging (we already average per-domain
    # by stacking both domains with equal size-agnostic weighting; use 1/n per row)
    if lam_d is None:
        lam_d = _corr_lambda(Z_aug, y_aug, off_aug, fam, frac)
    lam_w = c * lam_d
    pen = np.concatenate([np.full(p, lam_w), np.full(p, lam_d)])
    beta = fit_penalized(Z_aug, y_aug, off_aug, lam_d, family=fam, lam2=pen)
    delta = beta[p:2 * p]
    omega = beta[:p]
    return np.abs(delta), dict(lam_d=lam_d, lam_w=lam_w, omega=omega)


def method_sgk(D, lam=None, frac=0.08, B=5, family=None, rng=None):
    """SGShift-K (Eq. 5-6): knockoffs. Returns aggregated |W_j| importance."""
    if rng is None:
        rng = np.random.default_rng(12345)
    fam = family or D["family"]
    Z = D["Xs_T"]
    offset = D["hS_T"]
    y = D["Y_T"]
    if lam is None:
        lam = _corr_lambda(Z, y, offset, fam, frac)
    p = Z.shape[1]
    Wagg = np.zeros(p)
    for b in range(B):
        Xtil = gaussian_knockoffs(Z, rng)
        Zk = np.hstack([Z, Xtil])
        beta = fit_penalized(Zk, y, offset, lam, family=fam)
        d0 = beta[:p]
        dt = beta[p:2 * p]
        Wj = np.sign(np.abs(d0) - np.abs(dt)) * np.maximum(np.abs(d0), np.abs(dt))
        Wagg += np.abs(Wj)
    Wagg /= B
    return Wagg, dict(lam=lam, B=B)


def method_sgka(D, lam_d=None, frac=0.08, c=0.3, B=5, family=None, rng=None):
    """SGShift-KA: knockoffs + absorption. Target-specific delta+knockoff with a
    shared omega block fit on both domains."""
    if rng is None:
        rng = np.random.default_rng(12345)
    fam = family or D["family"]
    ZS, ZT = D["Xs_S"], D["Xs_T"]
    yS, yT = D["Y_S"], D["Y_T"]
    offS, offT = D["hS_S"], D["hS_T"]
    p = ZS.shape[1]
    nS, nT = ZS.shape[0], ZT.shape[0]
    # blocks: omega (shared) on [ZS; ZT]; delta (target) on [0; ZT];
    #         delta-knockoff on [0; Xtil_T]
    if lam_d is None:
        # lambda sized on the TARGET-ONLY correction design (so delta has the same
        # effective capacity as plain SGShift / SGShift-K)
        lam_d = _corr_lambda(ZT, yT, offT, fam, frac)
    lam_w = c * lam_d
    Xtil = gaussian_knockoffs(ZT, rng)  # one draw for the augmented design
    # augmented columns: [omega(ZS;ZT), delta(0;ZT), delta_k(0;Xtil)]
    omega_block = np.vstack([ZS, ZT])
    delta_block = np.vstack([np.zeros((nS, p)), ZT])
    deltak_block = np.vstack([np.zeros((nS, p)), Xtil])
    Z_aug = np.hstack([omega_block, delta_block, deltak_block])
    off_aug = np.concatenate([offS, offT])
    y_aug = np.concatenate([yS, yT])
    pen = np.concatenate([np.full(p, lam_w), np.full(p, lam_d), np.full(p, lam_d)])
    beta = fit_penalized(Z_aug, y_aug, off_aug, lam_d, family=fam, lam2=pen, max_iter=1200)
    delta = beta[p:2 * p]
    deltak = beta[2 * p:3 * p]
    Wj = np.sign(np.abs(delta) - np.abs(deltak)) * np.maximum(np.abs(delta), np.abs(deltak))
    return np.abs(Wj), dict(lam_d=lam_d, lam_w=lam_w)


# ---- baselines ------------------------------------------------------------- #
def _fit_target_model(D, family):
    """Separately trained target base model h_T (for Diff / WhyShift / SHAP)."""
    Z = D["Xs_T"]
    y = D["Y_T"]
    beta = fit_penalized(Z, y, np.zeros(Z.shape[0]), 1e-3, family=family, max_iter=1500)
    return beta


def method_diff(D, family=None):
    """Diff baseline: fit h_T separately, compute d = h_T - h_S on target, then
    sparse-regress d on X (Eq. from Sec.4 'Baselines')."""
    fam = family or D["family"]
    Z = D["Xs_T"]
    y = D["Y_T"]
    hT_beta = _fit_target_model(D, fam)
    hT = Z @ hT_beta
    d = hT - D["hS_T"]
    lam = 0.1 * _lam_max(Z, d, np.zeros(Z.shape[0]), "linear")
    gamma = fit_penalized(Z, d, np.zeros(Z.shape[0]), lam, family="linear")
    return np.abs(gamma), dict(lam=lam)


def method_shap(D, family=None):
    """SHAP-difference baseline (adapted from Mougan et al., Sec.4): fit h_S and
    h_T separately, compute per-feature |mean|SHAP|| difference. For linear base
    models SHAP_j(i) = coef_j*(x_ij - mean_j); the magnitude reduces to
    |coef_j| * mean|x_ij - mean_j|  -> difference isolates the shift contribution."""
    fam = family or D["family"]
    Z = D["Xs_T"]
    mu = Z.mean(axis=0)
    hS_beta = D["hS_beta"]
    hT_beta = _fit_target_model(D, fam)
    # mean absolute SHAP per feature (baseline = feature mean)
    mS = np.abs(hS_beta) * np.mean(np.abs(Z - mu), axis=0)
    mT = np.abs(hT_beta) * np.mean(np.abs(Z - mu), axis=0)
    score = np.abs(mT - mS)
    return score, dict()


def method_whyshift(D, family=None, max_depth=4, min_leaf=20):
    """WhyShift baseline (Liu et al. 2023): fit h_S, h_T separately; compute model
    discrepancy d = h_T - h_S on target; train a decision tree on d; features
    appearing on any split (importance>0) are the flagged shifted features. We use
    Gini/impurity-reduction importance as the ranking score."""
    fam = family or D["family"]
    Z = D["Xs_T"]
    hT_beta = _fit_target_model(D, fam)
    hT = Z @ hT_beta
    d = hT - D["hS_T"]
    imp = _cart_importance(Z, d, max_depth=max_depth, min_leaf=min_leaf)
    return imp, dict()


def _cart_importance(X, y, max_depth=4, min_leaf=20):
    n, p = X.shape
    imp = np.zeros(p)
    # iterative node stack: (indices, depth)
    stack = [(np.arange(n), 0)]
    while stack:
        idx, depth = stack.pop()
        if depth >= max_depth or len(idx) < 2 * min_leaf:
            continue
        yv = y[idx]
        parent_var = yv.var()
        if parent_var < 1e-12:
            continue
        best = None
        for j in range(p):
            xj = X[idx, j]
            # try a few split thresholds (quantiles)
            qs = np.quantile(xj, np.linspace(0.2, 0.8, 5))
            for thr in qs:
                left = idx[xj <= thr]
                right = idx[xj > thr]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                wL = len(left) / len(idx)
                wR = len(right) / len(idx)
                gain = parent_var - (wL * y[left].var() + wR * y[right].var())
                if best is None or gain > best[0]:
                    best = (gain, j, thr, left, right)
        if best is None or best[0] <= 0:
            continue
        gain, j, thr, left, right = best
        imp[j] += gain * (len(idx) / n)
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    return imp


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def score_all(D, methods=("sg", "sga", "sgk", "sgka", "diff", "whyshift", "shap"),
              seeds_knock=(0,), family=None):
    """Return dict method -> attribution score vector (abs importance)."""
    fam = family or D["family"]
    rng = np.random.default_rng(7)
    out = {}
    for m in methods:
        if m == "sg":
            s, _ = method_sg(D, family=fam)
        elif m == "sga":
            s, _ = method_sga(D, family=fam)
        elif m == "sgk":
            s, _ = method_sgk(D, family=fam, rng=rng)
        elif m == "sgka":
            s, _ = method_sgka(D, family=fam, rng=rng)
        elif m == "diff":
            s, _ = method_diff(D, family=fam)
        elif m == "whyshift":
            s, _ = method_whyshift(D, family=fam)
        elif m == "shap":
            s, _ = method_shap(D, family=fam)
        else:
            raise ValueError(m)
        out[m] = s
    return out


def evaluate(D, methods=("sg", "sga", "sgk", "sgka", "diff", "whyshift", "shap"),
             fpr=0.05, family=None):
    """Compute AUC and recall@FPR for each method on dataset D."""
    scores = score_all(D, methods=methods, family=family)
    labels = np.zeros(D["p"])
    labels[D["shift_set"]] = 1
    res = {}
    for m, s in scores.items():
        res[m] = dict(auc=auc_score(labels, s), recall=auc_recall_helper(labels, s, fpr))
    return res


def auc_recall_helper(labels, s, fpr):
    return recall_at_fpr(labels, s, fpr)


def run_experiment(cfg, seeds, methods=("sg", "sga", "sgk", "sgka", "diff", "whyshift", "shap"),
                   fpr=0.05, family=None):
    """Run cfg over seeds; return aggregated per-method mean AUC / recall and CIs."""
    cfg = dict(cfg)
    fam = family or cfg.pop("family", "logistic")
    cfg.pop("seed", None)
    auc_acc = {m: [] for m in methods}
    rec_acc = {m: [] for m in methods}
    for sd in seeds:
        D = gen_dataset(family=fam, **cfg, seed=sd)
        res = evaluate(D, methods=methods, fpr=fpr, family=family)
        for m in methods:
            auc_acc[m].append(res[m]["auc"])
            rec_acc[m].append(res[m]["recall"])
    summary = {}
    for m in methods:
        a = np.array(auc_acc[m])
        r = np.array(rec_acc[m])
        summary[m] = dict(
            auc_mean=float(np.nanmean(a)),
            auc_ci=float(1.96 * np.nanstd(a) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            recall_mean=float(np.nanmean(r)),
            recall_ci=float(1.96 * np.nanstd(r) / np.sqrt(len(r))) if len(r) > 1 else 0.0,
        )
    return summary


def auc_recall_helper(labels, s, fpr):
    return recall_at_fpr(labels, s, fpr)


def elbow_recovery(D, ranking="sgka", family=None, k_list=None):
    """Claim 4 / Sec.4.2 mechanism: cumulative target-domain performance recovery
    as a function of the number of top-ranked features included in the SGShift
    correction. With a SPARSE concept shift, recovery rises sharply then plateaus
    (an 'elbow'), i.e. a small feature subset recovers most of the performance loss.
    Returns dict with k_list, recovery fraction, and the k needed for 90% recovery.
    """
    fam = family or D["family"]
    methods_score = score_all(D, methods=[ranking], family=fam)
    score = methods_score[ranking]
    p = D["p"]
    order = np.argsort(-score, kind="mergesort")
    yT = D["Y_T"]
    off = D["hS_T"]
    L0 = logistic_loss(yT, off) if fam == "logistic" else 0.5 * np.mean((yT - off) ** 2)
    Zfull = D["Xs_T"]
    lam_full = _corr_lambda(Zfull, yT, off, fam, 0.08)
    bf = fit_penalized(Zfull, yT, off, lam_full, family=fam)
    Lfull = logistic_loss(yT, off + Zfull @ bf) if fam == "logistic" else 0.5 * np.mean((yT - (off + Zfull @ bf)) ** 2)
    if k_list is None:
        k_list = list(range(1, p + 1, max(1, p // 20))) + [p]
    rec = []
    for k in k_list:
        cols = order[:k]
        Zk = Zfull[:, cols]
        bk = fit_penalized(Zk, yT, off, lam_full, family=fam)
        Lk = logistic_loss(yT, off + Zk @ bk) if fam == "logistic" else 0.5 * np.mean((yT - (off + Zk @ bk)) ** 2)
        frac = (L0 - Lk) / (L0 - Lfull + 1e-12)
        rec.append(float(np.clip(frac, 0, 1.2)))
    rec = np.array(rec)
    k90 = int(k_list[int(np.argmax(rec >= 0.9))]) if np.any(rec >= 0.9) else p
    return dict(ranking=ranking, k_list=k_list, recovery=rec.tolist(),
                k_for_90pct=k90, p=p,
                recovery_at_3rd=k90 < p / 3)


def diabetes_config(p=33, n_S=None, n_T=None, rho=0.25, shift_mag=1.0, mismatch=False, seed=0):
    """Faithful semi-synthetic analog of the Diabetes 30-Day Readmission benchmark
    (73,615 samples, 33 features, source=non-ER / target=ER). Raw UCI/TableShift
    data is not available in this environment, so feature marginals are simulated
    (Gaussian AR(1)); the SHIFT STRUCTURE (sparse concept shift on a subset of
    features, identical marginal P(X) across domains) is preserved, which is what
    the method must exploit. Returns a gen_dataset config dict."""
    return dict(p=p, n_S=n_S or 30000, n_T=n_T or 15000, rho=rho, shift_mag=shift_mag,
                mismatch=mismatch, seed=seed,
                source_active=list(range(10)), shift_set=list(range(20, 25)))


if __name__ == "__main__":
    cfg = dict(p=30, n_S=2000, n_T=2000, rho=0.3, shift_mag=1.0, mismatch=False)
    seeds = list(range(12))
    s = run_experiment(cfg, seeds)
    for m in s:
        print(f"{m:10s} AUC={s[m]['auc_mean']:.3f}  recall@5%={s[m]['recall_mean']:.3f}")
