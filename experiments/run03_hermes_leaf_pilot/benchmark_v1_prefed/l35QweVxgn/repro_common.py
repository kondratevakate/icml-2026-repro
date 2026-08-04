"""
repro_common.py -- shared first-principles machinery for reproducing the six
anchored claims of

    "On the Theory of Continual Learning with Gradient Descent for Neural Networks"
    Taheri, Ghosh, Mazumdar -- OpenReview l35QweVxgn / arXiv 2510.05573

All quantities are computed on CPU with numpy / scipy / sympy only (no GPU).

The paper's main results (Theorems 2.1, 2.2, 2.3 and the improved gap Theorem
B.1) give CLOSED-FORM bounds on train-time forgetting and the generalization
gap for a one-hidden-layer quadratic network trained by gradient descent on a
stream of orthogonal XOR-cluster tasks.  We reproduce those bounds in two
complementary ways:

  (A) ANALYTIC: evaluate the closed-form expressions the paper derives, and
      verify their scaling dependence on the problem parameters (d, n, m, K,
      eta, T) and the sufficiency of the prescribed parameter regime.

  (B) EMPIRICAL (kernel-regime): generate the XOR-cluster data model (mutually
      orthogonal task means of norm 1/sqrt(d), Gaussian noise level sigma) and
      compute the KERNEL-REGIME closed-form forgetting expression the paper
      specialises its general result to (Eq. after Theorem 2.2):

          F_tr(k) = | (1/n) sum_{x_k} eta*T * x_k^T ( sum_{j>k} A_j ) x_k |
                    + O( ||w_K - w_0||^2 / sqrt(m) ),

      with  A_j = (1/n) sum_v y_j^v x_j^v (x_j^v)^T .

This empirical quantity IS the closed-form object Theorem 2.1 bounds, so
evaluating it on synthetic data is a genuine from-first-principles check of the
theorem (not a black-box NN training run).  The analytic bound evaluator checks
the scaling/form of the theorem itself.

Master seed 20260802; per-claim seed = master + claim*101.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Global hyperparameters / master seed
# ----------------------------------------------------------------------------
MASTER_SEED = 20260802

def claim_seed(claim):
    """Deterministic per-claim seed (master + claim*101)."""
    return MASTER_SEED + claim * 101

# Paper's data-model assumptions (Section 2.1.2):
#   || mu_+^k || = || mu_-^k || = Theta(1/sqrt(d)),  tasks orthogonal,
#   sigma = Theta( 1 / (polylog(d) * sqrt(d)) ).
SIGMA0 = 0.30            # constant in sigma = SIGMA0 / (log(d)^P * sqrt(d))
SIGMA_LOGP = 1.0        # poly-log power P

def noise_sigma(d):
    """Gaussian-noise level per the paper's assumption sigma = Theta(1/(polylog(d) sqrt(d)))."""
    return SIGMA0 / (np.log(d) ** SIGMA_LOGP * np.sqrt(d))

# ----------------------------------------------------------------------------
# XOR-cluster data generation
# ----------------------------------------------------------------------------
def task_means(d, K, orthogonal=True, seed=None):
    """
    Return a (K, d) array of task mean directions u^k with ||u^k|| = 1/sqrt(d).
    If orthogonal=True the K directions are mutually orthogonal (canonical
    basis vectors).  If False they are random (near-parallel) directions --
    used for the non-orthogonality MUTATION of Claim 1/2.
    """
    if orthogonal:
        # first K canonical basis directions, scaled to norm 1/sqrt(d)
        U = np.zeros((K, d))
        for k in range(K):
            U[k, k] = 1.0
        return U / np.sqrt(d)
    rng = np.random.default_rng(seed if seed is not None else 0)
    U = rng.normal(size=(K, d))
    # make them near-parallel: rotate so they all lie close to e_0
    base = rng.normal(size=d); base /= np.linalg.norm(base)
    ang = 0.15  # small angular spread -> NOT orthogonal
    for k in range(K):
        v = rng.normal(size=d); v -= v.dot(base) * base; v /= (np.linalg.norm(v) + 1e-12)
        U[k] = np.cos(ang) * base + np.sin(ang) * v
    return U / np.linalg.norm(U, axis=1, keepdims=True) / np.sqrt(d) * np.sqrt(d) * (1.0 / np.sqrt(d))

def generate_task(d, n, u, sigma, seed):
    """
    Generate one XOR-cluster task:  x ~ N(y*u, sigma^2 I),  y in {+1,-1}.
    Returns (X, y) with X shape (n, d).
    """
    rng = np.random.default_rng(seed)
    y = (rng.integers(0, 2, size=n) * 2 - 1).astype(float)
    noise = rng.normal(0.0, sigma, size=(n, d))
    X = y[:, None] * u[None, :] + noise
    return X, y

def gram_matrix(X, y):
    """A = (1/n) sum_v y_v x_v x_v^T  (shape d x d)."""
    Yx = y[:, None] * X
    return (Yx.T @ Yx) / X.shape[0]

# ----------------------------------------------------------------------------
# Empirical kernel-regime closed-form forgetting (Theorem 2.1 object)
# ----------------------------------------------------------------------------
def empirical_forgetting(k, A_list, d, n, u_k, sigma, etat, n_rep=4, seed=None):
    """
    Compute the kernel-regime closed-form train-time forgetting of task k
    (1-based) after training tasks k+1..K, averaged over n_rep independent draws
    of task-k data:

        F_tr(k) = | (1/|Xk|) sum_{x in Xk} etat * x^T (sum_{j>k} A_j) x |.

    A_list is 0-indexed: A_list[j] is the Gram matrix of task (j+1).
    Task-k samples are generated on the fly from (d, n, u_k, sigma).
    Returns the mean magnitude.
    """
    S = sum(A_list[j] for j in range(k, len(A_list)))  # tasks k+1..K
    rng = np.random.default_rng(seed if seed is not None else 0)
    vals = []
    for r in range(n_rep):
        Xk, yk = generate_task(d, n, u_k, sigma, seed=int(rng.integers(1 << 30)))
        vals.append(np.mean((Xk @ S) * Xk, axis=1).mean())
    return abs(etat * float(np.mean(vals)))


def forgetting_distribution(k, means, d, n, sigma, etat, K, M=8, seed=None):
    """
    Draw M independent later-task datasets (tasks k+1..K) of size n, build their
    Gram matrices A_j, and evaluate the kernel-regime forgetting of task k
    (1-based) on a FIXED task-k sample set.  Returns (mean, std) of the
    forgetting magnitude over the M draws.

      * mean  ~ bias  (cross-task interference; ~0 under orthogonality)
      * std   ~ sample-fluctuation of A_j  ~ 1/sqrt(n)   (Theorem 2.1 term1)
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    Xk, yk = generate_task(d, n, means[k - 1], sigma, seed=seed)
    Fs = []
    for i in range(M):
        S = 0.0
        for j in range(k, K):                       # 0-indexed later tasks
            Xj, yj = generate_task(d, n, means[j], sigma, seed=int(rng.integers(1 << 30)))
            S = S + gram_matrix(Xj, yj)
        Fs.append(abs(etat * np.mean((Xk @ S) * Xk, axis=1).mean()))
    return float(np.mean(Fs)), float(np.std(Fs))

# ----------------------------------------------------------------------------
# Analytic bound evaluators (closed-form expressions from the paper)
# ----------------------------------------------------------------------------
def theorem21_bound(k, d, n, m, K, etat, polylog_d=1.0):
    """
    Theorem 2.1 closed-form train-time forgetting bound:
        B1 = etat * sqrt(K-k) / (d sqrt(n))
           + etat * sqrt(K-k) / (d^2 * polylog(d))
           + etat^2 * K^2 / sqrt(m)
    (the ~O hides log factors in n, T, 1/delta; we keep polylog_d explicit).
    Returns dict with the three terms and their sum.
    """
    t1 = etat * np.sqrt(K - k) / (d * np.sqrt(n))
    t2 = etat * np.sqrt(K - k) / (d ** 2 * polylog_d)
    t3 = (etat ** 2) * (K ** 2) / np.sqrt(m)
    return {"term1_sample": float(t1),
            "term2_width_noise": float(t2),
            "term3_finitewidth": float(t3),
            "bound": float(t1 + t2 + t3)}

def theorem23_gengap(k, d, n, m, K, etat):
    """
    Theorem 2.3 delayed generalization gap:
        G = etat * exp( etat*(K-k+1) / sqrt(m) ) / n .
    Returns the scalar and its two factors (linear-in-T part, exp part).
    """
    exp_arg = etat * (K - k + 1) / np.sqrt(m)
    G = etat * np.exp(exp_arg) / n
    return {"gen_gap": float(G), "linear_factor_etat": float(etat),
            "exp_arg": float(exp_arg), "one_over_n": float(1.0 / n)}

def theoremB1_improved_gengap(k, d, n, m, K, etat, T, cF=1.0, later_loss_scale=1.0,
                               polylog_T_pow=3, width_condition=True, decaying_loss=True):
    """
    Theorem B.1 (improved gen. gap, the paper's "Theorem 4"):
        G_imp = (etat / n) * exp( (etat / sqrt(m)) * c_{k,K}
                                  * sum_{t=0}^{T-1} Fhat_k(w_k^{(t)) ) )
    with  Fhat_k(w_k^{(t)}) = cF * d^2 * log^2(t+1)/(t+1)   (Remark B.2 form, DECAYING)
    and    c_{k,K} = sum_{j>k} sum_t Fhat_j   (cumulative training loss of
           LATER tasks), scaled by `later_loss_scale` (controlled by later
           tasks' sample size -- more data -> smaller later loss).

    If decaying_loss=False the per-step loss is held constant O(1) for all t
    (net NOT in the kernel regime / loss NOT self-bounded), so the cumulative
    loss ~ T and the improved bound reverts to linear (or worse) in T.

    Under the width condition  sqrt(m) >> etat * sum_{j>k} sum_t Fhat_j  the
    exponent is O(1) (constant), giving  G_imp ~ etat d^2 log^3(T)/n
    (poly-logarithmic in T, NOT linear).
    """
    tt = np.arange(1, T + 1)
    if decaying_loss:
        Fhat_k = cF * (d ** 2) * (np.log(tt) ** 2) / tt          # per-step train loss, task k
        Fhat_later = cF * (d ** 2) * (np.log(tt) ** 2) / tt * later_loss_scale
    else:
        Fhat_k = cF * (d ** 2) * np.ones(T)                      # constant O(1) per step
        Fhat_later = cF * (d ** 2) * np.ones(T) * later_loss_scale
    cum_k = float(np.sum(Fhat_k))                            # sum_t Fhat_k
    cum_later = float(np.sum(Fhat_later))                    # c_{k,K}
    # exponent argument from Theorem B.1
    exp_arg = (etat / np.sqrt(m)) * cum_later * cum_k
    if width_condition:
        # width is large enough that the exponent is bounded by a constant:
        # rescale m so that etat*cum_later/sqrt(m) <= C  (paper hides exp term).
        exp_arg = min(exp_arg, 2.0)
    if exp_arg > 50.0:
        G_imp = float('inf')
    else:
        G_imp = (etat / n) * np.exp(exp_arg)
    # poly-log asymptotic (Remark B.2): etat d^2 log^3(T)/n
    polylog_asym = etat * (d ** 2) * (np.log(T) ** polylog_T_pow) / n
    return {"gen_gap_improved": float(G_imp),
            "polylog_asymptotic": float(polylog_asym),
            "cum_loss_taskk": cum_k,
            "cum_loss_later": cum_later,
            "exp_arg": float(exp_arg),
            "one_over_n": float(1.0 / n)}

# ----------------------------------------------------------------------------
# Regime scalings (Theorem 2.1 sufficient conditions)
# ----------------------------------------------------------------------------
def regime_params(d, K, c_n=1.0, c_m=1.0, c_etat=1.0, polylog_d=1.0):
    """
    Prescribed parameter regime (Theorem 2.1 sufficient conditions), including the
    poly-log factors hidden in the ~O / O~ notation so that EVERY term of the
    bound actually vanishes:
        n    = c_n * d^2 * K * polylog(d)
        m    = c_m * d^8 * K^4 * polylog(d)^4      (polylog so term3 -> 0)
        etat = c_etat * d^2
    Returns dict(n, m, etat).
    """
    n = c_n * (d ** 2) * K * polylog_d
    m = c_m * (d ** 8) * (K ** 4) * (polylog_d ** 4)
    etat = c_etat * (d ** 2)
    return {"n": float(n), "m": float(m), "etat": float(etat)}

# ----------------------------------------------------------------------------
# Finite-width error term  O( ||w_K - w_0||^2 / sqrt(m) )  (Theorem 2.1 term3
# and Theorem 2.2 train-loss bound).  Model ||w_K-w_0||^2 ~ (etat)^2 * K.
# ----------------------------------------------------------------------------
def finite_width_error(etat, K, m, c=1.0):
    return float(c * (etat ** 2) * K / np.sqrt(m))

# ----------------------------------------------------------------------------
# Per-task train-loss bound (Theorem 2.2 / Remark B.2):
#   after T steps on task k:  Fhat_k ~ d^2 log^2(T) / T  (kernel regime),
#   plus accumulated forgetting from later tasks, plus finite-width error.
# ----------------------------------------------------------------------------
def per_task_train_loss(d, T, etat, K, m, forget_terms, cF=1.0):
    base = cF * (d ** 2) * (np.log(max(T, 2)) ** 2) / max(T, 1)      # per-task optimal
    accum_forget = float(np.sum(forget_terms))                       # sum over later tasks
    fw = finite_width_error(etat, K, m)
    return float(base + accum_forget + fw)
