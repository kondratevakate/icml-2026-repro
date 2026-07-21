"""
Reproduction (Phase 1) of the CONFORMAL CORE of:
  "CalPro: Prior-Aware Evidential-Conformal Prediction ... for Protein Structures"
  Shihab, Akter, Sharma. ICML 2026 (orid 3LRWjJTp0Y), arXiv 2601.07201.

Goal of Phase 1: understand conformal prediction hands-on and set up CalPro's
claim 1 (coverage degradation under distribution shift). We use the paper's OWN
non-biological setting (p.13): a heteroscedastic regression with high-noise
regions at extreme covariate ranges, evaluated under covariate shift. No proteins,
no AlphaFold, no docking (that is claim 3, Phase 2). numpy only, CPU.

Split conformal (paper Eq. 4, Eq. 9): with nonconformity scores s_i = |y_i - mu(x_i)|
on a calibration set, pick a radius q_hat so that intervals [mu(x) - q_hat, mu(x) + q_hat]
cover the true y with probability >= tau on EXCHANGEABLE data (finite-sample guarantee).
CalPro's point is that vanilla split conformal keeps coverage in-distribution but its
coverage DEGRADES under shift; their evidential + adaptive machinery (Phase 2) restores it.

>>> Kate implements `split_conformal_interval` below — it is the heart of the method. <<<
"""
import numpy as np

RNG_SEED = 0


# --------------------------- data: heteroscedastic regression ---------------------------
def true_mean(x):
    # smooth nonlinear ground-truth signal g(x)
    return np.sin(1.5 * x) + 0.5 * x


def noise_scale(x):
    # heteroscedastic: noise grows at extreme covariate ranges (paper's "high-noise regimes")
    return 0.1 + 0.6 * (np.abs(x) ** 1.5)


def sample_data(n, x_low, x_high, rng):
    """y = g(x) + sigma(x) * eps, x ~ Uniform[x_low, x_high]."""
    x = rng.uniform(x_low, x_high, size=n)
    y = true_mean(x) + noise_scale(x) * rng.standard_normal(n)
    return x, y


# --------------------------- base predictor: closed-form least squares ------------------
def poly_features(x, degree=3):
    return np.vstack([x ** d for d in range(degree + 1)]).T  # [1, x, x^2, x^3]


def fit_base_predictor(x, y, degree=3, ridge=1e-6):
    """Deliberately simple point predictor mu(x). Conformal wraps ANY predictor."""
    Phi = poly_features(x, degree)
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    w = np.linalg.solve(A, Phi.T @ y)
    return w


def predict(w, x, degree=3):
    return poly_features(x, degree) @ w


# ============================ >>> YOUR PART (the conformal core) <<< ====================
def split_conformal_interval(cal_scores, tau):
    """
    Split-conformal calibration -- the finite-sample heart of conformal prediction.

    Given nonconformity scores on a held-out CALIBRATION set and a target coverage
    level tau in (0, 1), return the conformal radius q_hat such that the intervals
        [ mu(x) - q_hat,  mu(x) + q_hat ]
    achieve AT LEAST tau marginal coverage on exchangeable (calibration <-> test) data.

    Args:
        cal_scores : 1D np.ndarray of nonconformity scores s_i = |y_i - mu(x_i)|
                     on the calibration set (length n_cal).
        tau        : target coverage level in (0, 1), e.g. 0.90.

    Returns:
        q_hat : float -- the interval radius (the conformal quantile).

    THE ONE IDEA: do NOT return the plain tau-empirical-quantile of cal_scores. The
    finite-sample guarantee needs the (n_cal + 1)-corrected level: take the k-th SMALLEST
    score with k = ceil((n_cal + 1) * tau). If k > n_cal the guarantee is vacuous at this
    tau -> return np.inf. (Vovk et al.; Lei et al. 2018. The "+1" is the whole trick: it
    accounts for the unseen test point, giving marginal coverage >= tau exactly.)
    """
    n = cal_scores.shape[0]
    k = int(np.ceil((n + 1) * tau))          # rank of the corrected quantile (1-indexed)
    if k > n:                                 # not enough calibration points for this tau
        return np.inf
    return float(np.sort(cal_scores)[k - 1])  # k-th smallest score = conformal radius
# =======================================================================================


# ------------------------------- coverage evaluation harness ----------------------------
def empirical_coverage(y_true, mu, q_hat):
    """Fraction of points whose true y falls inside [mu - q_hat, mu + q_hat]."""
    return float(np.mean(np.abs(y_true - mu) <= q_hat))


def run(degree=3, n_train=2000, n_cal=2000, n_test=4000, taus=(0.8, 0.9, 0.95)):
    rng = np.random.default_rng(RNG_SEED)
    # in-distribution: x in [-2, 2]
    xtr, ytr = sample_data(n_train, -2.0, 2.0, rng)
    xcal, ycal = sample_data(n_cal, -2.0, 2.0, rng)
    xte, yte = sample_data(n_test, -2.0, 2.0, rng)          # in-distribution test
    # shifted: x in [2, 3.5] -- extreme covariate range, higher noise (covariate shift)
    xsh, ysh = sample_data(n_test, 2.0, 3.5, rng)

    w = fit_base_predictor(xtr, ytr, degree)
    cal_scores = np.abs(ycal - predict(w, xcal, degree))     # s_i = |y - mu|

    print(f"{'tau':>5} {'q_hat':>8} {'cov_indist':>11} {'cov_shift':>10}   claim-1 signature")
    for tau in taus:
        q = split_conformal_interval(cal_scores, tau)
        cov_id = empirical_coverage(yte, predict(w, xte, degree), q)
        cov_sh = empirical_coverage(ysh, predict(w, xsh, degree), q)
        degr = cov_id - cov_sh
        flag = "coverage DROPS under shift" if degr > 0.03 else "holds"
        print(f"{tau:5.2f} {q:8.3f} {cov_id:11.3f} {cov_sh:10.3f}   {flag} (drop {degr:+.3f})")
    print("\nExpected: in-distribution coverage ~ nominal tau; under shift it degrades.")
    print("That degradation IS CalPro's claim 1 baseline; Phase 2 (evidential head) restores it.")


def selftest():
    """Once split_conformal_interval is implemented: in-distribution coverage ~ tau."""
    rng = np.random.default_rng(1)
    x, y = sample_data(6000, -2.0, 2.0, rng)
    w = fit_base_predictor(x[:2000], y[:2000])
    cal = np.abs(y[2000:4000] - predict(w, x[2000:4000]))
    te_mu = predict(w, x[4000:]); te_y = y[4000:]
    ok = True
    for tau in (0.8, 0.9, 0.95):
        q = split_conformal_interval(cal, tau)
        cov = empirical_coverage(te_y, te_mu, q)
        good = cov >= tau - 0.03
        ok &= good
        print(f"  selftest tau={tau}: coverage={cov:.3f}  {'OK' if good else 'LOW (check +1 correction)'}")
    print("SELFTEST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    import sys
    try:
        if "--selftest" in sys.argv:
            selftest()
        else:
            run()
    except NotImplementedError as e:
        print(f"[!] {e}")
        print("    Open verify_calpro.py, fill split_conformal_interval, then rerun.")
