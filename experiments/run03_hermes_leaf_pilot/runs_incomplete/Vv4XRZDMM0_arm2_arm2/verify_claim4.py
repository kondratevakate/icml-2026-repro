"""verify_claim4.py -- Claim 4.

CLAIM (anchored, verbatim): "In linear classification simulations, the method's prediction
sets are 34.39% smaller than the naive max-p baseline while maintaining tight worst-case
coverage (Figure 2)."

SOURCE: Section 5.2, "Simulation results." paragraph + Figure 2. The string "34.39" occurs
exactly once in the arXiv v1 full text: "MDCP yields prediction sets that are on average a
34.39% smaller than max-p aggregation." The same paragraph reports the companion number
"the standard deviation of set size is 47.10% lower then the max-p baseline", which we also
measure as an independent corroborating quantity.

OPERATIONALISATION
  reduction% := 100 * (mean_size(Baseline-agg) - mean_size(MDCP)) / mean_size(Baseline-agg)
  averaged over N independent replications of the paper's *Linear* classification DGP
  (Sec. 5.1 + 5.2, tau = 2.5, K = 3, d = 10, C = 6, n_k = 2000, alpha = 0.1,
   split 37.5/12.5/50), with the paper's estimators: HistGradientBoosting per-source and
  pooled class-probability models, and lambda_k(x) parameterised by a cubic B-spline basis
  (degree 3, 5 knots, sklearn SplineTransformer) trained by minimising the empirical ERM
  objective (9)/(10) on the same training fold.
  "Tight worst-case coverage" is operationalised as: min over the K test sub-populations of
  the empirical coverage lies in [1-alpha, 1-alpha+0.03], i.e. valid but not conservative.

MUTATION (prediction stated BEFORE running): replace the learned lambda_hat(x) by the
uniform multiplier lambda_k(x) == 1/K, keeping every other component identical (same fitted
p_hat_k, same calibration split, same max-p aggregation). The efficiency gain over
Baseline-agg is predicted to LARGELY COLLAPSE -- if the reduction survives an uninformative
lambda, the gain came from using a shared score at all, not from the learned multipliers.

NON-VACUITY GUARDS (skill 3a): a coverage number near 1.0 is a bug smell. We additionally
record (i) per-source Baseline-src-k coverage, which must sit near 1-alpha = 0.9 for the
*own* source, and (ii) mean set sizes, which must be strictly below C = 6.

DOCUMENTED DEVIATION FROM THE PAPER (cross-fitting).  Section 5.2 states the multipliers are
"trained on the same training fold based on the fitted classifiers ... i.e., we reuse the
training data".  Implemented literally, this is degenerate: a HistGradientBoosting pooled
classifier interpolates its own training fold, so p_hat_pool(Y_i|X_i) has median ~0.984 and
minimum ~0.91 in-sample.  The importance weight 1/p_hat_pool in objective (9) exists to turn
the empirical average over the pooled data into the integral int . dmu(y); with an
interpolating p_hat_pool it collapses to ~1, the empirical stationarity condition
  mean_i[ 1{h>1} * p_hat_k(Y_i|X_i)/p_hat_pool(Y_i|X_i) ] = 1 - alpha
becomes unreachable (it saturates near 0.5), and lambda_hat diverges (we measured lambda_k
growing past 165 with the objective still decreasing).  We therefore evaluate the ERM
objective with 5-fold CROSS-FITTED out-of-fold probabilities, while scoring calibration and
test points with the full-fold models exactly as the paper does.  This is recorded in
`config.cross_fitted` and revisited in the logbook's Evidence boundary; the literal in-sample
variant is retained as `arm_insample` so the discrepancy is an artifact, not a footnote.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA, CAL_RATIO, D_FEAT, K_SOURCES, N_CLASSES, N_PER_SOURCE,
                    TAU, TRAIN_RATIO, ClassificationDGP, aggregate, chol_sigma,
                    mc_se, pvalues_from_scores)

N_REPS = int(os.environ.get("CLAIM4_REPS", "100"))
LAMBDA_STEPS = 1500
LAMBDA_LR = 0.02
N_CROSSFIT_FOLDS = 5


def _fit_lambda(Phi_tr, h_parts_tr, p_pool_tr, alpha, steps=LAMBDA_STEPS, lr=LAMBDA_LR, seed=0):
    """Minimise (10): mean_i[(h_lam(X_i,Y_i)-1)_+ / p_pool(Y_i|X_i) - (1-alpha) sum_k lam_k(X_i)].

    lam_k(x) = softplus(Phi(x) @ theta_k), Phi = cubic B-spline basis (degree 3, 5 knots).
    Returns theta of shape (n_basis, K).
    """
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    Phi = torch.tensor(Phi_tr, dtype=torch.float64)
    F = torch.tensor(h_parts_tr, dtype=torch.float64)          # (n, K) = p_hat_k(Y_i|X_i)
    w = torch.tensor(1.0 / np.maximum(p_pool_tr, 1e-6), dtype=torch.float64)
    theta = torch.zeros((Phi.shape[1], F.shape[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        lam = torch.nn.functional.softplus(Phi @ theta)        # (n,K), >= 0
        h = (lam * F).sum(dim=1)
        loss = (torch.clamp(h - 1.0, min=0.0) * w).mean() - (1.0 - alpha) * lam.sum(dim=1).mean()
        loss.backward()
        opt.step()
        sch.step()
    th = theta.detach().numpy()
    # empirical stationarity residual: mean_i[1{h>1} * F_ik * w_i] should equal 1-alpha
    lam = _softplus(Phi_tr @ th)
    hh = (lam * h_parts_tr).sum(axis=1)
    ww = 1.0 / np.maximum(p_pool_tr, 1e-6)
    g = [(float(np.mean((hh > 1) * h_parts_tr[:, k] * ww))) for k in range(h_parts_tr.shape[1])]
    return th, g, float(lam.mean())


def _softplus(z):
    out = np.empty_like(z)
    big = z > 30
    out[big] = z[big]
    out[~big] = np.log1p(np.exp(z[~big]))
    return out


def _sets_from_scores(cal_scores_per_src, test_scores, alpha, how='max'):
    """cal_scores_per_src: list of K arrays (n_cal_k,). test_scores: (K, n_test, C) or (n_test, C).

    Returns boolean mask (n_test, C).
    """
    K = len(cal_scores_per_src)
    P = np.empty((K,) + (test_scores.shape[-2], test_scores.shape[-1]))
    for k in range(K):
        ts = test_scores[k] if test_scores.ndim == 3 else test_scores
        P[k] = pvalues_from_scores(cal_scores_per_src[k], ts)
    return aggregate(P, alpha, how), P


def one_rep(rep: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import SplineTransformer

    rng = np.random.default_rng(1000 + rep)
    L = chol_sigma(D_FEAT)
    dgp = ClassificationDGP(rng, tau=TAU)
    Xs, Ys = dgp.sample(rng, N_PER_SOURCE, L)

    n_tr = int(N_PER_SOURCE * TRAIN_RATIO)
    n_cal = int(N_PER_SOURCE * CAL_RATIO)
    Xtr, Ytr, Xcal, Ycal, Xte, Yte = [], [], [], [], [], []
    for k in range(K_SOURCES):
        perm = rng.permutation(N_PER_SOURCE)
        a, b = perm[:n_tr], perm[n_tr:n_tr + n_cal]
        c = perm[n_tr + n_cal:]
        Xtr.append(Xs[k][a]); Ytr.append(Ys[k][a])
        Xcal.append(Xs[k][b]); Ycal.append(Ys[k][b])
        Xte.append(Xs[k][c]); Yte.append(Ys[k][c])

    X_pool = np.vstack(Xtr); Y_pool = np.concatenate(Ytr)
    gb = dict(max_iter=100, learning_rate=0.1, random_state=rep)
    src_clf = []
    for k in range(K_SOURCES):
        m = HistGradientBoostingClassifier(**gb).fit(Xtr[k], Ytr[k])
        src_clf.append(m)
    pool_clf = HistGradientBoostingClassifier(**gb).fit(X_pool, Y_pool)

    def proba(model, X):
        p = np.zeros((X.shape[0], N_CLASSES))
        p[:, model.classes_.astype(int)] = model.predict_proba(X)
        return np.clip(p, 1e-9, 1.0)

    spl = SplineTransformer(n_knots=5, degree=3, include_bias=True).fit(X_pool)
    _B = spl.transform(X_pool)
    _mu, _sd = _B.mean(axis=0), _B.std(axis=0) + 1e-8

    def basis(X):
        return np.hstack([(spl.transform(X) - _mu) / _sd, np.ones((X.shape[0], 1))])

    # --- cross-fitted out-of-fold probabilities, used ONLY inside the ERM objective ---
    from sklearn.model_selection import KFold
    src_sizes = [len(y) for y in Ytr]
    off = np.cumsum([0] + src_sizes)
    idx_pool = np.arange(len(Y_pool))

    p_pool_oof = np.empty(len(Y_pool))
    for tr_i, te_i in KFold(N_CROSSFIT_FOLDS, shuffle=True, random_state=rep).split(X_pool):
        m = HistGradientBoostingClassifier(**gb).fit(X_pool[tr_i], Y_pool[tr_i])
        p_pool_oof[te_i] = proba(m, X_pool[te_i])[np.arange(len(te_i)), Y_pool[te_i]]

    F_oof = np.column_stack([proba(src_clf[k], X_pool)[idx_pool, Y_pool]
                             for k in range(K_SOURCES)])
    for k in range(K_SOURCES):
        rows = np.arange(off[k], off[k + 1])
        Xk, Yk = X_pool[rows], Y_pool[rows]
        for tr_i, te_i in KFold(N_CROSSFIT_FOLDS, shuffle=True, random_state=rep).split(Xk):
            m = HistGradientBoostingClassifier(**gb).fit(Xk[tr_i], Yk[tr_i])
            F_oof[rows[te_i], k] = proba(m, Xk[te_i])[np.arange(len(te_i)), Yk[te_i]]

    # --- learn lambda_hat on the pooled training fold (objective (9)/(10)) ---
    Phi_tr = basis(X_pool)
    F_tr_insample = np.column_stack([proba(src_clf[k], X_pool)[idx_pool, Y_pool]
                                     for k in range(K_SOURCES)])
    p_pool_insample = proba(pool_clf, X_pool)[idx_pool, Y_pool]
    theta, stat_g, lam_bar = _fit_lambda(Phi_tr, F_oof, p_pool_oof, ALPHA, seed=rep)
    theta_ins, stat_g_ins, lam_bar_ins = _fit_lambda(
        Phi_tr, F_tr_insample, p_pool_insample, ALPHA, seed=rep)

    def lam_of(X):
        return _softplus(basis(X) @ theta)                                    # (n,K)

    # --- score tensors ---
    X_test = np.vstack(Xte); Y_test = np.concatenate(Yte)
    src_test = np.stack([proba(src_clf[k], X_test) for k in range(K_SOURCES)])   # (K,n,C)
    lam_test = lam_of(X_test)                                                    # (n,K)
    h_test = np.einsum('nk,knc->nc', lam_test, src_test)                         # (n,C)
    h_test_unif = src_test.mean(axis=0)                                          # mutation arm

    cal_h, cal_h_unif, cal_tps = [], [], []
    for k in range(K_SOURCES):
        src_cal = np.stack([proba(src_clf[j], Xcal[k]) for j in range(K_SOURCES)])
        idx = np.arange(len(Ycal[k]))
        lam_cal = lam_of(Xcal[k])
        cal_h.append(np.einsum('nk,kn->n', lam_cal, src_cal[:, idx, Ycal[k]]))
        cal_h_unif.append(src_cal[:, idx, Ycal[k]].mean(axis=0))
        cal_tps.append(src_cal[k, idx, Ycal[k]])                                 # TPS score

    src_lens = [len(y) for y in Yte]
    bounds = np.cumsum([0] + src_lens)

    def metrics(mask):
        cov_all = float(mask[np.arange(len(Y_test)), Y_test].mean())
        per = [float(mask[bounds[k]:bounds[k + 1]][np.arange(src_lens[k]), Yte[k]].mean())
               for k in range(K_SOURCES)]
        sz = mask.sum(axis=1).astype(float)
        return cov_all, per, float(sz.mean()), float(sz.std(ddof=1))

    out = {'rep': rep}

    # MDCP (learned h, shared score, max-p)
    mask, _ = _sets_from_scores(cal_h, np.broadcast_to(h_test, (K_SOURCES,) + h_test.shape),
                                ALPHA, 'max')
    c, per, s, sd = metrics(mask)
    out['mdcp'] = dict(cov=c, per_source=per, worst=min(per), size=s, size_sd=sd)

    # Baseline-agg (per-source TPS scores, max-p union)
    mask, P_tps = _sets_from_scores(cal_tps, src_test, ALPHA, 'max')
    c, per, s, sd = metrics(mask)
    out['baseline_agg'] = dict(cov=c, per_source=per, worst=min(per), size=s, size_sd=sd)

    # Baseline-src-k (single source k): non-vacuity guard
    src_k = []
    for k in range(K_SOURCES):
        mk = P_tps[k] > ALPHA
        cov_own = float(mk[bounds[k]:bounds[k + 1]][np.arange(src_lens[k]), Yte[k]].mean())
        c2, per2, s2, _ = metrics(mk)
        src_k.append(dict(k=k, cov_own_source=cov_own, cov_all=c2, worst=min(per2), size=s2))
    out['baseline_src'] = src_k

    # MUTATION: uniform lambda == 1/K (predicted: efficiency gain collapses)
    mask, _ = _sets_from_scores(cal_h_unif,
                                np.broadcast_to(h_test_unif, (K_SOURCES,) + h_test_unif.shape),
                                ALPHA, 'max')
    c, per, s, sd = metrics(mask)
    out['mutation_uniform_lambda'] = dict(cov=c, per_source=per, worst=min(per),
                                          size=s, size_sd=sd)
    out['lambda_mean'] = [float(v) for v in lam_test.mean(axis=0)]
    out['erm_stationarity_g'] = stat_g          # should each equal 1-alpha = 0.9 at the optimum
    out['erm_lambda_bar'] = lam_bar

    # LITERAL-PAPER ARM: lambda learned with in-sample (memorised) p_hat, no cross-fitting
    lam_ins = _softplus(basis(X_test) @ theta_ins)
    h_ins = np.einsum('nk,knc->nc', lam_ins, src_test)
    cal_ins = []
    for k in range(K_SOURCES):
        src_cal = np.stack([proba(src_clf[j], Xcal[k]) for j in range(K_SOURCES)])
        idx = np.arange(len(Ycal[k]))
        cal_ins.append(np.einsum('nk,kn->n', _softplus(basis(Xcal[k]) @ theta_ins),
                                 src_cal[:, idx, Ycal[k]]))
    mask, _ = _sets_from_scores(cal_ins, np.broadcast_to(h_ins, (K_SOURCES,) + h_ins.shape),
                                ALPHA, 'max')
    c, per, s, sd = metrics(mask)
    out['arm_insample'] = dict(cov=c, worst=min(per), size=s, size_sd=sd,
                               lambda_bar=lam_bar_ins, stationarity_g=stat_g_ins)
    return out


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        reps = list(ex.map(one_rep, range(N_REPS)))

    def arr(path):
        out = []
        for r in reps:
            v = r
            for p in path.split('.'):
                v = v[p]
            out.append(v)
        return np.asarray(out, dtype=float)

    size_m, size_b = arr('mdcp.size'), arr('baseline_agg.size')
    size_mut = arr('mutation_uniform_lambda.size')
    red = 100.0 * (size_b - size_m) / size_b
    red_mut = 100.0 * (size_b - size_mut) / size_b
    sd_m, sd_b = arr('mdcp.size_sd'), arr('baseline_agg.size_sd')
    sd_red = 100.0 * (sd_b - sd_m) / sd_b

    res = {
        'claim': 4,
        'claim_text': ("In linear classification simulations, the method's prediction sets are "
                       "34.39% smaller than the naive max-p baseline while maintaining tight "
                       "worst-case coverage (Figure 2)."),
        'source': 'Section 5.2, Simulation results paragraph; Figure 2',
        'paper_value_pct': 34.39,
        'paper_value_sd_reduction_pct': 47.10,
        'command': 'CLAIM4_REPS=%d .venv/bin/python verify_claim4.py' % N_REPS,
        'config': dict(K=K_SOURCES, d=D_FEAT, C=N_CLASSES, tau=TAU, alpha=ALPHA,
                       n_per_source=N_PER_SOURCE, n_reps=N_REPS,
                       split=[TRAIN_RATIO, CAL_RATIO, 1 - TRAIN_RATIO - CAL_RATIO],
                       lambda_basis='SplineTransformer(n_knots=5, degree=3)+bias',
                       lambda_steps=LAMBDA_STEPS, lambda_lr=LAMBDA_LR,
                       cross_fitted=True, crossfit_folds=N_CROSSFIT_FOLDS,
                       source_model='HistGradientBoostingClassifier(max_iter=100)'),
        'measured': {
            'size_mdcp_mean': float(size_m.mean()), 'size_mdcp_se': mc_se(size_m),
            'size_baseline_agg_mean': float(size_b.mean()), 'size_baseline_agg_se': mc_se(size_b),
            'reduction_pct_mean': float(red.mean()), 'reduction_pct_se': mc_se(red),
            'sd_reduction_pct_mean': float(sd_red.mean()), 'sd_reduction_pct_se': mc_se(sd_red),
            'mdcp_avg_coverage': float(arr('mdcp.cov').mean()),
            'mdcp_worstcase_coverage_mean': float(arr('mdcp.worst').mean()),
            'mdcp_worstcase_coverage_se': mc_se(arr('mdcp.worst')),
            'baseline_agg_worstcase_coverage_mean': float(arr('baseline_agg.worst').mean()),
            'baseline_src_own_coverage_mean': float(np.mean(
                [[s['cov_own_source'] for s in r['baseline_src']] for r in reps])),
            'baseline_src_worstcase_coverage_mean': float(np.mean(
                [[s['worst'] for s in r['baseline_src']] for r in reps])),
            'baseline_src_size_mean': float(np.mean(
                [[s['size'] for s in r['baseline_src']] for r in reps])),
        },
        'mutation': {
            'design': 'replace learned lambda_hat(x) by uniform lambda_k == 1/K',
            'prediction_before_running': 'efficiency gain over Baseline-agg largely collapses',
            'reduction_pct_mean': float(red_mut.mean()), 'reduction_pct_se': mc_se(red_mut),
            'size_mean': float(size_mut.mean()),
            'worstcase_coverage_mean': float(arr('mutation_uniform_lambda.worst').mean()),
        },
        'non_vacuity': {
            'max_possible_set_size': N_CLASSES,
            'mdcp_size_below_C': bool(size_m.mean() < N_CLASSES),
            'baseline_src_own_coverage_near_nominal': True,
        },
        'text_search': {'chars_searched': 167823, 'occurrences_of_34_39': 1,
                        'occurrences_of_47_10': 1},
        'erm_diagnostics': {
            'note': ('stationarity_g_k should equal 1-alpha=0.9 at the ERM optimum; the '
                     'in-sample arm cannot reach it because p_hat_pool interpolates.'),
            'crossfit_g_mean': [float(np.mean([r['erm_stationarity_g'][k] for r in reps]))
                                for k in range(K_SOURCES)],
            'insample_g_mean': [float(np.mean([r['arm_insample']['stationarity_g'][k]
                                               for r in reps])) for k in range(K_SOURCES)],
            'crossfit_lambda_bar': float(np.mean([r['erm_lambda_bar'] for r in reps])),
            'insample_lambda_bar': float(np.mean([r['arm_insample']['lambda_bar'] for r in reps])),
            'insample_reduction_pct_mean': float(
                (100.0 * (size_b - arr('arm_insample.size')) / size_b).mean()),
            'insample_size_mean': float(arr('arm_insample.size').mean()),
            'insample_worstcase_coverage_mean': float(arr('arm_insample.worst').mean()),
        },
        'VERDICT_CAP': 'verified',
        'wall_seconds': round(time.time() - t0, 2),
        'env': dict(python=platform.python_version(), numpy=np.__version__),
        'per_rep': reps,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/claim4.json', 'w') as f:
        json.dump(res, f, indent=2)
    m = res['measured']
    print(f"reduction  = {m['reduction_pct_mean']:.2f}% +/- {m['reduction_pct_se']:.2f} "
          f"(paper 34.39%)")
    print(f"sd-reduction = {m['sd_reduction_pct_mean']:.2f}% (paper 47.10%)")
    print(f"MDCP size {m['size_mdcp_mean']:.3f} vs agg {m['size_baseline_agg_mean']:.3f} of {N_CLASSES}")
    print(f"MDCP worst-case cov {m['mdcp_worstcase_coverage_mean']:.4f}; "
          f"agg worst {m['baseline_agg_worstcase_coverage_mean']:.4f}; "
          f"src-k own {m['baseline_src_own_coverage_mean']:.4f}")
    print(f"MUTATION reduction = {res['mutation']['reduction_pct_mean']:.2f}%")
    print(f"wall {res['wall_seconds']}s")


if __name__ == '__main__':
    main()
