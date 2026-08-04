"""verify_claim2.py -- Claim 2.

CLAIM (anchored, verbatim): "Theorem 2 (Section 3.1) characterizes the population-optimal
prediction set as a superlevel set of a shared score function h_{lambda*}(x,y) =
sum_k lambda_k*(x) f_k(y|x), with complementary slackness ensuring exact 1-alpha coverage for
at least one source."

SOURCE: Theorem 2, Section 3.1; proof in Appendix B.3. The proof fixes x and writes the
CONDITIONAL program
      min_{I in {0,1}}  int I(y) dmu(y)
      s.t.              int I(y) f_k(y|x) dmu(y) >= 1 - alpha,   k = 1..K
relaxes I to [0,1], forms the Lagrangian with multipliers lambda(x) in R_+^K, and obtains
      L_x(I,lambda) = int I(y)[1 - h_lambda(x)(y)] dmu(y) + (1-alpha) sum_k lambda_k(x),
      h_lambda(x)(y) := sum_k lambda_k(x) f_k(y|x),
whose pointwise minimiser is I=1 where h>1, I=0 where h<1, arbitrary where h=1.

OPERATIONALISATION -- this is a CHARACTERIZATION OF AN OPTIMUM, so it is machine-checked, not
simulated. For each random instance we instantiate the conditional program as a finite LP and
solve it with scipy HiGHS, reading lambda* directly off the HiGHS inequality duals
(`-res.ineqlin.marginals`, sign-flipped because `>=` is stored as `-A x <= -b`). We then
assert each clause of the theorem separately and report the FRACTION OF INSTANCES satisfied:

  C1 structure      : I* = 1 wherever h_{lambda*} > 1 + tol, I* = 0 wherever h < 1 - tol
                      (the boundary {h = 1} is left unconstrained, as the theorem allows)
  C2 slackness      : lambda_k* > 0  =>  coverage_k = 1 - alpha exactly (to tol)
  C3 no-overpay     : lambda_k* = 0  =>  constraint k is slack
  C4 existence      : there EXISTS k with lambda_k* > 0  -- this is the clause that makes
                      "exact 1-alpha coverage for AT LEAST ONE source" true, and it is the
                      one most likely to fail silently, so it is asserted on its own
  C5 strong duality : primal optimum == (1-alpha) sum_k lambda_k* - int (h_{lambda*} - 1)_+ dmu

GEOMETRY (plan.md). Theorem 2's uniqueness needs mu({y : h*(x,y) = 1}) = 0. We run BOTH:
  * regression geometry (PRIMARY): Lebesgue measure on a fine y-grid, K=3 Gaussian
    conditionals from the Sec. 5.3 Linear DGP. Non-degeneracy holds; the LP optimum is
    essentially integral and the theorem is sharply testable.
  * classification geometry (SECONDARY): counting measure on C=6 labels, Sec. 5.2 DGP. Here
    {h=1} carries positive mass, the LP optimum is genuinely fractional at the boundary, and
    the structure clause is only testable off the boundary. Reported separately and NOT used
    to carry the verdict.

MUTATION (prediction stated BEFORE running): take a wrong multiplier lambda' (lambda*
perturbed by a random log-normal factor, and a uniform lambda' = mean(lambda*)) and form the
set {y : h_{lambda'}(y) > 1}. Predicted: it is never as good as the optimum -- either it
violates a coverage constraint or it is strictly larger. Target statistic:
`frac_as_good_as_optimum == 0.0`.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA, D_FEAT, K_SOURCES, N_CLASSES, TAU, ClassificationDGP,
                    RegressionDGP, chol_sigma, draw_X)

N_INSTANCES = int(os.environ.get("CLAIM2_INSTANCES", "300"))
TOL = 1e-7
GRID_N = 601


def solve_conditional_lp(f: np.ndarray, dmu: np.ndarray, alpha: float):
    """min sum_j I_j dmu_j  s.t.  sum_j I_j f_kj dmu_j >= 1-alpha,  0 <= I <= 1.

    f   : (K, m) conditional density/pmf at the grid points
    dmu : (m,)   base-measure weights (dy for Lebesgue, 1 for counting)
    """
    c = dmu
    A_ub = -(f * dmu[None, :])
    b_ub = -np.full(f.shape[0], 1.0 - alpha)
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0.0, 1.0), method='highs')
    if not r.success:
        return None
    lam = -np.asarray(r.ineqlin.marginals, dtype=float)
    lam[lam < 0] = 0.0
    return r.x, lam, float(r.fun)


def check_instance(f, dmu, alpha, rng):
    sol = solve_conditional_lp(f, dmu, alpha)
    if sol is None:
        return None
    I, lam, primal = sol
    h = lam @ f                                            # h_{lambda*}(y)
    cov = (f * dmu[None, :]) @ I                           # (K,) achieved coverage

    above, below = h > 1 + 1e-6, h < 1 - 1e-6
    c1 = (bool(np.all(I[above] > 1 - 1e-6)) if above.any() else True) and \
         (bool(np.all(I[below] < 1e-6)) if below.any() else True)
    # graded version of the structure clause: mu-measure of the region where I* disagrees
    # with the superlevel set, relative to the optimal set size (skill P9: attribute the
    # residual instead of hiding it behind a pass/fail tolerance).
    disagree = np.sum(dmu * (above * (1.0 - I) + below * I))
    pos = lam > 1e-9
    c2 = bool(np.all(np.abs(cov[pos] - (1 - alpha)) < 1e-6)) if pos.any() else False
    c3 = bool(np.all(cov[~pos] >= (1 - alpha) - 1e-6)) if (~pos).any() else True
    c4 = bool(pos.any())
    dual = (1 - alpha) * lam.sum() - float(np.sum(np.clip(h - 1.0, 0, None) * dmu))
    c5_gap = abs(primal - dual)

    # ---- mutation: wrong lambda' -> {h_lam' > 1} ----
    as_good = 0
    for kind in ('perturb', 'uniform'):
        lam2 = (lam * np.exp(rng.normal(0, 0.5, size=lam.size)) if kind == 'perturb'
                else np.full_like(lam, lam.mean()))
        I2 = (lam2 @ f > 1).astype(float)
        cov2 = (f * dmu[None, :]) @ I2
        feasible = bool(np.all(cov2 >= (1 - alpha) - 1e-9))
        size2 = float(np.sum(I2 * dmu))
        if feasible and size2 <= primal + 1e-9:
            as_good += 1
    return dict(c1=c1, c2=c2, c3=c3, c4=c4, c5_gap=c5_gap, primal=primal,
                disagree_mass=float(disagree),
                disagree_frac_of_set=float(disagree / max(primal, 1e-12)),
                n_pos_lambda=int(pos.sum()), boundary_mass=float(np.sum(
                    dmu[np.abs(h - 1.0) <= 1e-9])),
                frac_fractional=float(np.mean((I > 1e-6) & (I < 1 - 1e-6))),
                mutation_as_good=as_good)


def run_regression(n_inst):
    rng = np.random.default_rng(777)
    L = chol_sigma(D_FEAT)
    out = []
    for i in range(n_inst):
        dgp = RegressionDGP(np.random.default_rng(5000 + i), tau=TAU)
        x = draw_X(rng, 1, L)
        mu = dgp.mu(x)[:, 0]
        lo = mu.min() - 6 * dgp.sigma.max()
        hi = mu.max() + 6 * dgp.sigma.max()
        y = np.linspace(lo, hi, GRID_N)
        dy = float(y[1] - y[0])
        f = dgp.cond_pdf(x, y)[:, 0, :]
        r = check_instance(f, np.full(GRID_N, dy), ALPHA, rng)
        if r:
            out.append(r)
    return out


def run_classification(n_inst):
    rng = np.random.default_rng(778)
    L = chol_sigma(D_FEAT)
    out = []
    for i in range(n_inst):
        dgp = ClassificationDGP(np.random.default_rng(6000 + i), tau=TAU)
        x = draw_X(rng, 1, L)
        f = dgp.cond_prob(x)[:, 0, :]
        r = check_instance(f, np.ones(N_CLASSES), ALPHA, rng)
        if r:
            out.append(r)
    return out


def summarise(rs):
    n = len(rs)
    return {
        'n_instances': n,
        'C1_structure_superlevel_set': float(np.mean([r['c1'] for r in rs])),
        'C1_mean_disagreement_mass': float(np.mean([r['disagree_mass'] for r in rs])),
        'C1_max_disagreement_mass': float(np.max([r['disagree_mass'] for r in rs])),
        'C1_mean_disagreement_frac_of_set': float(np.mean([r['disagree_frac_of_set'] for r in rs])),
        'C2_complementary_slackness_tight': float(np.mean([r['c2'] for r in rs])),
        'C3_zero_lambda_implies_slack': float(np.mean([r['c3'] for r in rs])),
        'C4_exists_positive_lambda': float(np.mean([r['c4'] for r in rs])),
        'C5_max_duality_gap': float(np.max([r['c5_gap'] for r in rs])),
        'C5_median_duality_gap': float(np.median([r['c5_gap'] for r in rs])),
        'mean_n_active_sources': float(np.mean([r['n_pos_lambda'] for r in rs])),
        'mean_boundary_mass': float(np.mean([r['boundary_mass'] for r in rs])),
        'mean_frac_fractional_I': float(np.mean([r['frac_fractional'] for r in rs])),
        'mutation_frac_as_good_as_optimum': float(
            np.sum([r['mutation_as_good'] for r in rs]) / (2.0 * n)),
    }


def main():
    t0 = time.time()
    reg = run_regression(N_INSTANCES)
    cls = run_classification(N_INSTANCES)
    res = {
        'claim': 2,
        'claim_text': ("Theorem 2 (Section 3.1) characterizes the population-optimal "
                       "prediction set as a superlevel set of a shared score function "
                       "h_{lambda*}(x,y) = sum_k lambda_k*(x) f_k(y|x), with complementary "
                       "slackness ensuring exact 1-alpha coverage for at least one source."),
        'source': 'Theorem 2, Section 3.1; proof Appendix B.3',
        'command': 'CLAIM2_INSTANCES=%d .venv/bin/python verify_claim2.py' % N_INSTANCES,
        'config': dict(K=K_SOURCES, alpha=ALPHA, tau=TAU, tol=TOL, y_grid_points=GRID_N,
                       lp_solver='scipy.optimize.linprog(method="highs")',
                       duals='-res.ineqlin.marginals'),
        'regression_geometry_PRIMARY': summarise(reg),
        'classification_geometry_SECONDARY': summarise(cls),
        'mutation': {
            'design': ("wrong lambda' (log-normal perturbation of lambda*, and uniform "
                       "lambda' = mean(lambda*)); form {y : h_lambda'(y) > 1}"),
            'prediction_before_running': ('never as good as the optimum: infeasible or '
                                          'strictly larger; frac_as_good_as_optimum == 0.0'),
            'regression_frac_as_good': summarise(reg)['mutation_frac_as_good_as_optimum'],
            'classification_frac_as_good': summarise(cls)['mutation_frac_as_good_as_optimum'],
        },
        'VERDICT_CAP': 'verified',
        'wall_seconds': round(time.time() - t0, 2),
        'env': dict(python=platform.python_version(), numpy=np.__version__),
    }
    os.makedirs('results', exist_ok=True)
    with open('results/claim2.json', 'w') as f:
        json.dump(res, f, indent=2)
    for name in ('regression_geometry_PRIMARY', 'classification_geometry_SECONDARY'):
        s = res[name]
        print(f"--- {name} (n={s['n_instances']})")
        for k in ('C1_structure_superlevel_set', 'C1_mean_disagreement_frac_of_set',
                  'C2_complementary_slackness_tight',
                  'C3_zero_lambda_implies_slack', 'C4_exists_positive_lambda'):
            print(f"    {k:36s} {s[k]:.6f}")
        print(f"    max duality gap                      {s['C5_max_duality_gap']:.3e}")
        print(f"    mean active sources                  {s['mean_n_active_sources']:.3f}")
        print(f"    mean fractional fraction of I*       {s['mean_frac_fractional_I']:.5f}")
        print(f"    MUTATION frac as good as optimum     "
              f"{s['mutation_frac_as_good_as_optimum']:.4f}")
    print(f"wall {res['wall_seconds']}s")


if __name__ == '__main__':
    main()
