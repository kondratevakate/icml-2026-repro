"""verify_claim3.py -- Claim 3.

CLAIM (anchored, verbatim): "Theorem 3 (Section 3.2) proves max-p aggregation with learned
conformity scores is asymptotically optimal, converging to the oracle-optimal prediction set
up to boundary regions."

SOURCE: Theorem 3, Section 3.2; proof in Appendix B.4. Statement as printed:
  ASSUME sup_x ||lambda_hat(x) - lambda*(x)||_inf ->p 0 and
         sup_{x,y} |f_hat_k(y|x) - f_k(y|x)| ->p 0, with sup_x ||lambda_hat||_inf <= M;
  THEN   sup_{x,y} |h_hat(x,y) - h*(x,y)| ->p 0, and with T := {(x,y) : h*(x,y) = 1},
         limsup_n rho( C_hat^(n)  triangle  {(x,y) : h*(x,y) >= 1} )  <=  rho(T).
The "up to boundary regions" in the anchored claim is exactly the rho(T) slack.

OPERATIONALISATION
  Regression geometry (plan.md): Sec. 5.3 Linear DGP, K=3 Gaussian conditionals, Lebesgue
  measure on a fine y-grid. Here rho(T) = 0 (the level set {h* = 1} is a finite set of grid
  crossings, Lebesgue-null), so the theorem's bound reduces to
        rho(C_hat^(n) triangle C*)  ->  0,
  which is a sharply falsifiable statement -- this is why the classification/counting geometry
  is NOT used for this claim (there rho(T) > 0 and the claim is not sharply testable).

  * C* = {y : h*(x,y) >= 1}, with lambda*(x) obtained EXACTLY by solving the conditional LP
    of Theorem 2 with scipy HiGHS and reading off the duals (same machinery as claim 2).
  * C_hat^(n) = the max-p aggregated set built from the estimated score h_hat, calibrated on
    n points per source -- i.e. the deployed Algorithm 1, not an oracle shortcut.
  * The theorem's PREMISE is instantiated with estimation error eps_n = n^(-1/2):
        lambda_hat_k(x) = max(0, lambda*_k(x) * (1 + eps_n * z_k(x)))
        f_hat_k(y|x)    = max(0, f_k(y|x) + eps_n * xi_k(x) * scale)
    so sup-norm errors vanish at the standard rate, satisfying the hypothesis as written.
  * Measured statistic: rho_n := mean over test x of int |1{y in C_hat^(n)} - 1{y in C*}| dy.

MUTATION -- violate the theorem's PREMISE, not its conclusion (prediction stated BEFORE
running). Feed non-converging scores through the IDENTICAL estimator and the IDENTICAL
n-sweep:
    arm "uniform"      : lambda_hat_k(x) == mean_k lambda*_k(x)   (never converges to lambda*)
    arm "single_source": lambda_hat = lambda*_1(x) * e_1          (wrong source only)
  Predicted: rho_n PLATEAUS instead of decaying. A plateau is only visible as a trend, so the
  entire sweep is reported, together with the log-log decay slope (expected ~negative for the
  converging arm, ~0 for the mutation arms) and the shrink factor across the sweep.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA, D_FEAT, K_SOURCES, TAU, RegressionDGP, aggregate,
                    chol_sigma, draw_X, mc_se, pvalues_from_scores)
from verify_claim2 import solve_conditional_lp

N_REPS = int(os.environ.get("CLAIM3_REPS", "12"))
N_GRID = [125, 250, 500, 1000, 2000]
N_TEST_X = 20
GRID_N = 401
ARMS = ('converging', 'uniform', 'single_source')


def _lam_star(dgp, X, ygrid, dy):
    """Exact lambda*(x) from the Theorem-2 conditional LP, one solve per x."""
    f_all = dgp.cond_pdf(X, ygrid)                       # (K, n, m)
    dmu = np.full(ygrid.size, dy)
    out = np.zeros((X.shape[0], K_SOURCES))
    for i in range(X.shape[0]):
        sol = solve_conditional_lp(f_all[:, i, :], dmu, ALPHA)
        if sol is not None:
            out[i] = sol[1]
    return out, f_all


def _lam_hat(lam_star, arm, eps, rng):
    if arm == 'converging':
        z = rng.standard_normal(lam_star.shape)
        return np.maximum(0.0, lam_star * (1.0 + eps * z))
    if arm == 'uniform':
        return np.repeat(lam_star.mean(axis=1, keepdims=True), K_SOURCES, axis=1)
    if arm == 'single_source':
        out = np.zeros_like(lam_star)
        out[:, 0] = lam_star[:, 0]
        return out
    raise ValueError(arm)


def one_rep(rep: int):
    rng = np.random.default_rng(31000 + rep)
    L = chol_sigma(D_FEAT)
    dgp = RegressionDGP(np.random.default_rng(41000 + rep), tau=TAU)

    n_max = max(N_GRID)
    Xcal = [draw_X(rng, n_max, L) for _ in range(K_SOURCES)]
    Ycal = [dgp.b[k] + Xcal[k] @ dgp.beta[k] + dgp.sigma[k] * rng.standard_normal(n_max)
            for k in range(K_SOURCES)]
    Xte = draw_X(rng, N_TEST_X, L)

    mu_all = dgp.mu(np.vstack([Xte] + Xcal))
    lo = mu_all.min() - 6 * dgp.sigma.max()
    hi = mu_all.max() + 6 * dgp.sigma.max()
    ygrid = np.linspace(lo, hi, GRID_N)
    dy = float(ygrid[1] - ygrid[0])

    lam_te, f_te = _lam_star(dgp, Xte, ygrid, dy)                     # (nT,K), (K,nT,m)
    h_star_te = np.einsum('nk,knm->nm', lam_te, f_te)
    C_star = (h_star_te >= 1.0).astype(float)
    rho_T = float(np.mean(np.sum((np.abs(h_star_te - 1.0) < 1e-9) * dy, axis=1)))

    lam_cal = [_lam_star(dgp, Xcal[k], ygrid, dy)[0] for k in range(K_SOURCES)]
    # f_j(Y_i | X_i) at the calibration pairs of source k -- computed directly (never
    # materialise the (K, n, n) grid tensor).
    f_cal_true = []
    for k in range(K_SOURCES):
        mu_k = dgp.mu(Xcal[k])                                        # (K, n_max)
        z = (Ycal[k][None, :] - mu_k) / dgp.sigma[:, None]
        f_cal_true.append((np.exp(-0.5 * z ** 2) /
                           (np.sqrt(2 * np.pi) * dgp.sigma[:, None])).T)   # (n_max, K)

    fscale = float(np.mean(1.0 / (np.sqrt(2 * np.pi) * dgp.sigma)))
    res = {'rep': rep, 'rho_T': rho_T, 'sweep': {a: {} for a in ARMS}}
    for n in N_GRID:
        eps = n ** -0.5
        for arm in ARMS:
            # deterministic seed (hash() is salted per process -- never use it for seeding)
            r = np.random.default_rng([rep, n, ARMS.index(arm)])
            lh_te = _lam_hat(lam_te, arm, eps, r)
            f_te_hat = np.maximum(0.0, f_te + eps * fscale *
                                  r.standard_normal((K_SOURCES, N_TEST_X, 1)))
            h_te = np.einsum('nk,knm->nm', lh_te, f_te_hat)

            cal_h = []
            for k in range(K_SOURCES):
                lh_c = _lam_hat(lam_cal[k][:n], arm, eps, r)
                fc = np.maximum(0.0, f_cal_true[k][:n] + eps * fscale *
                                r.standard_normal((1, K_SOURCES)))
                cal_h.append(np.einsum('nk,nk->n', lh_c, fc))
            P = np.stack([pvalues_from_scores(cal_h[j], h_te) for j in range(K_SOURCES)])
            C_hat = aggregate(P, ALPHA, 'max').astype(float)
            rho = float(np.mean(np.sum(np.abs(C_hat - C_star) * dy, axis=1)))
            size_star = float(np.mean(np.sum(C_star * dy, axis=1)))
            res['sweep'][arm][str(n)] = dict(rho=rho, size_Cstar=size_star,
                                             size_Chat=float(np.mean(np.sum(C_hat * dy, axis=1))))
    return res


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        reps = list(ex.map(one_rep, range(N_REPS)))

    sweep = {}
    for arm in ARMS:
        rhos = np.array([[r['sweep'][arm][str(n)]['rho'] for n in N_GRID] for r in reps])
        m = rhos.mean(axis=0)
        sl = float(np.polyfit(np.log(N_GRID), np.log(np.maximum(m, 1e-12)), 1)[0])
        sweep[arm] = {
            'n_grid': N_GRID,
            'rho_mean': [float(v) for v in m],
            'rho_se': [mc_se(rhos[:, i]) for i in range(len(N_GRID))],
            'loglog_slope': sl,
            'loglog_slope_tail4': float(np.polyfit(np.log(N_GRID[1:]),
                                                   np.log(np.maximum(m[1:], 1e-12)), 1)[0]),
            'shrink_factor_first_to_last': float(m[0] / max(m[-1], 1e-12)),
            'shrink_factor_tail4': float(m[1] / max(m[-1], 1e-12)),
        }
    size_star = float(np.mean([r['sweep']['converging'][str(N_GRID[-1])]['size_Cstar']
                               for r in reps]))
    res = {
        'claim': 3,
        'claim_text': ("Theorem 3 (Section 3.2) proves max-p aggregation with learned "
                       "conformity scores is asymptotically optimal, converging to the "
                       "oracle-optimal prediction set up to boundary regions."),
        'source': 'Theorem 3, Section 3.2; proof Appendix B.4',
        'command': 'CLAIM3_REPS=%d .venv/bin/python verify_claim3.py' % N_REPS,
        'config': dict(K=K_SOURCES, d=D_FEAT, alpha=ALPHA, tau=TAU, n_reps=N_REPS,
                       n_grid=N_GRID, n_test_x=N_TEST_X, y_grid_points=GRID_N,
                       geometry='regression / Lebesgue (rho(T) = 0)',
                       eps_n='n^(-1/2)', lambda_star='exact, HiGHS duals of the Thm-2 LP'),
        'rho_T_measured': float(np.mean([r['rho_T'] for r in reps])),
        'mean_size_C_star': size_star,
        'sweep': sweep,
        'mutation': {
            'design': ('violate the premise: uniform lambda and single-source lambda fed '
                       'through the identical estimator and n-sweep'),
            'prediction_before_running': 'rho_n plateaus instead of decaying',
            'slopes': {a: sweep[a]['loglog_slope'] for a in ARMS},
            'slopes_tail4': {a: sweep[a]['loglog_slope_tail4'] for a in ARMS},
            'shrink_factors': {a: sweep[a]['shrink_factor_first_to_last'] for a in ARMS},
        },
        'residual_attribution': ('rho_n cannot reach 0: floors are the y-grid discretisation '
                                 '(dy), the finite number of test x, finite calibration, and '
                                 'MC noise across %d reps' % N_REPS),
        'VERDICT_CAP': 'verified',
        'wall_seconds': round(time.time() - t0, 2),
        'env': dict(python=platform.python_version(), numpy=np.__version__),
    }
    os.makedirs('results', exist_ok=True)
    with open('results/claim3.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f"rho(T) measured = {res['rho_T_measured']:.3e}   mean |C*| = {size_star:.3f}")
    for arm in ARMS:
        s = sweep[arm]
        print(f"{arm:14s} rho: " + "  ".join(f"{v:.4f}" for v in s['rho_mean']) +
              f"   slope {s['loglog_slope']:+.3f} (tail {s['loglog_slope_tail4']:+.3f})"
              f"  shrink {s['shrink_factor_first_to_last']:.2f}x (tail {s['shrink_factor_tail4']:.2f}x)")
    print(f"wall {res['wall_seconds']}s")


if __name__ == '__main__':
    main()
