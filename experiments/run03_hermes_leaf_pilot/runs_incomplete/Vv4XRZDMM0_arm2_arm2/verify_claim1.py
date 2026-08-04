"""verify_claim1.py -- Claim 1.

CLAIM (anchored, verbatim): "Theorem 1 (Section 2) establishes finite-sample uniform validity
of max-p aggregation, showing it yields coverage guarantees under any mixture distribution of
the K sources."

SOURCE: Theorem 1, Section 2; proof in Appendix B.1. The proof's core step is
  P(sup_j p^(j)(y) <= alpha) <= P(p^(k)(Y_{n+1}) <= alpha) <= alpha  under P^(k),
so coverage >= 1 - alpha holds under EVERY source k, hence under every mixture (coverage
under a mixture is the convex combination sum_k w_k * coverage_k, so the infimum over the
simplex is attained at a vertex -- the vertex sweep below is therefore EXHAUSTIVE over the
mixture space, not a sample of it).

OPERATIONALISATION
  Oracle classification DGP of Sec. 5.1/5.2 (Linear, tau=2.5, K=3, d=10, C=6, alpha=0.1) with
  the TRUE conditional probabilities f_k(y|x) -- Theorem 1 is a finite-sample statement that
  holds for ANY conformity score, so no model fitting is needed and none is done.
  Score: s_k(x,y) = -h(x,y) with h(x,y) = sum_k f_k(y|x) (an arbitrary, deliberately
  non-optimal shared score, to show validity does not depend on the score being good).
  p^(k)(y) = (1 + #{i in cal_k : h_i <= h(x,y)}) / (n_cal + 1),  C = {y : max_k p^(k) > alpha}.
  Finite sample: n_cal = 250 per source (the paper's 12.5% of 2000).
  Measured: coverage under each of the K vertices, and over an exhaustive 0.05-step grid of
  the 2-simplex of mixture weights (231 mixtures). Reported statistic: the MINIMUM over all
  mixtures of the mean coverage, which must be >= 1 - alpha.

MUTATION (aggregation ladder; prediction stated BEFORE running): replace `max` by `mean` and
by `min` in the aggregation rule, changing nothing else. Predicted:
  max  -> worst-case coverage >= 0.90  (Theorem 1 holds)
  mean -> worst-case coverage drops BELOW 0.90 (no uniform guarantee)
  min  -> collapses far below, toward the intersection of the per-source sets
A graded ladder is the evidence that it is the `max` operator specifically that buys validity.

NON-VACUITY GUARDS (skill 3a): coverage 1.0 is trivially achievable by returning all C=6
labels, so we also record (i) mean set size, which must be strictly < 6, and (ii) the
single-source coverage of each component set on its OWN source, which must sit near 0.9.
"""
from __future__ import annotations

import itertools
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA, D_FEAT, K_SOURCES, N_CLASSES, TAU, ClassificationDGP,
                    aggregate, chol_sigma, mc_se, pvalues_from_scores)

N_REPS = int(os.environ.get("CLAIM1_REPS", "200"))
N_CAL = 250
N_TEST = 1000
GRID_STEP = 0.05


def simplex_grid(K: int, step: float):
    m = int(round(1.0 / step))
    out = []
    for c in itertools.product(range(m + 1), repeat=K - 1):
        if sum(c) <= m:
            w = list(c) + [m - sum(c)]
            out.append(np.array(w, dtype=float) / m)
    return out


def one_rep(rep: int):
    rng = np.random.default_rng(20000 + rep)
    L = chol_sigma(D_FEAT)
    dgp = ClassificationDGP(rng, tau=TAU)

    # calibration + test data from each source (oracle densities: no training fold needed)
    cal_h, own_cov, per_src = [], [], {}
    Xc, Yc = dgp.sample(rng, N_CAL, L)
    Xt, Yt = dgp.sample(rng, N_TEST, L)

    for k in range(K_SOURCES):
        f = dgp.cond_prob(Xc[k])                                  # (K, n, C)
        idx = np.arange(N_CAL)
        cal_h.append(f[:, idx, Yc[k]].sum(axis=0))                # h = sum_k f_k

    res = {'rep': rep}
    for how in ('max', 'mean', 'min'):
        covs, sizes = [], []
        for k in range(K_SOURCES):
            f = dgp.cond_prob(Xt[k])                              # (K, n, C)
            h = f.sum(axis=0)                                     # (n, C)
            P = np.stack([pvalues_from_scores(cal_h[j], h) for j in range(K_SOURCES)])
            mask = aggregate(P, ALPHA, how)
            covs.append(float(mask[np.arange(N_TEST), Yt[k]].mean()))
            sizes.append(float(mask.sum(axis=1).mean()))
            if how == 'max':
                own = P[k] > ALPHA
                own_cov.append(float(own[np.arange(N_TEST), Yt[k]].mean()))
                per_src.setdefault('component_size', []).append(float(own.sum(axis=1).mean()))
        res[how] = dict(per_source_coverage=covs, worst=min(covs), mean_size=float(np.mean(sizes)))
    res['component_own_coverage'] = own_cov
    res['component_size'] = per_src['component_size']
    return res


def main():
    t0 = time.time()
    reps = [one_rep(r) for r in range(N_REPS)]
    grid = simplex_grid(K_SOURCES, GRID_STEP)

    out = {}
    for how in ('max', 'mean', 'min'):
        cov = np.array([r[how]['per_source_coverage'] for r in reps])       # (R, K)
        mean_cov = cov.mean(axis=0)                                         # (K,)
        se_cov = cov.std(axis=0, ddof=1) / np.sqrt(len(reps))
        mix = np.array([w @ mean_cov for w in grid])
        out[how] = {
            'per_source_coverage_mean': [float(v) for v in mean_cov],
            'per_source_coverage_se': [float(v) for v in se_cov],
            'worstcase_coverage': float(mean_cov.min()),
            'min_over_mixture_grid': float(mix.min()),
            'max_over_mixture_grid': float(mix.max()),
            'n_mixtures_enumerated': len(grid),
            'n_mixtures_below_nominal': int((mix < 1 - ALPHA).sum()),
            'mean_set_size': float(np.mean([r[how]['mean_size'] for r in reps])),
        }

    comp_cov = np.array([r['component_own_coverage'] for r in reps]).ravel()
    res = {
        'claim': 1,
        'claim_text': ("Theorem 1 (Section 2) establishes finite-sample uniform validity of "
                       "max-p aggregation, showing it yields coverage guarantees under any "
                       "mixture distribution of the K sources."),
        'source': 'Theorem 1, Section 2; proof Appendix B.1',
        'command': 'CLAIM1_REPS=%d .venv/bin/python verify_claim1.py' % N_REPS,
        'config': dict(K=K_SOURCES, d=D_FEAT, C=N_CLASSES, tau=TAU, alpha=ALPHA,
                       n_cal_per_source=N_CAL, n_test_per_source=N_TEST, n_reps=N_REPS,
                       simplex_grid_step=GRID_STEP, densities='oracle (no model fitting)',
                       score='h(x,y) = sum_k f_k(y|x), deliberately non-optimal'),
        'exhaustive_note': ('coverage under mixture w is sum_k w_k * coverage_k, so the '
                            'infimum over the simplex is attained at a vertex; the vertex '
                            'sweep is exhaustive and the grid is a redundant confirmation'),
        'arms': out,
        'mutation': {
            'design': 'aggregation ladder: max -> mean -> min, all else identical',
            'prediction_before_running': ('max >= 0.90; mean drops below 0.90; min collapses '
                                          'far below'),
            'ladder_worstcase': [out['max']['worstcase_coverage'],
                                 out['mean']['worstcase_coverage'],
                                 out['min']['worstcase_coverage']],
            'ladder_mean_size': [out['max']['mean_set_size'], out['mean']['mean_set_size'],
                                 out['min']['mean_set_size']],
        },
        'non_vacuity': {
            'max_possible_set_size': N_CLASSES,
            'maxp_mean_set_size': out['max']['mean_set_size'],
            'component_own_source_coverage_mean': float(comp_cov.mean()),
            'component_own_source_coverage_se': mc_se(comp_cov),
            'component_mean_size': float(np.mean([r['component_size'] for r in reps])),
        },
        'VERDICT_CAP': 'verified',
        'wall_seconds': round(time.time() - t0, 2),
        'env': dict(python=platform.python_version(), numpy=np.__version__),
    }
    os.makedirs('results', exist_ok=True)
    with open('results/claim1.json', 'w') as f:
        json.dump(res, f, indent=2)
    for how in ('max', 'mean', 'min'):
        o = out[how]
        print(f"{how:5s} worst-case cov {o['worstcase_coverage']:.4f}  "
              f"min-over-{o['n_mixtures_enumerated']}-mixtures {o['min_over_mixture_grid']:.4f}  "
              f"size {o['mean_set_size']:.3f}/{N_CLASSES}  "
              f"#mixtures<0.9: {o['n_mixtures_below_nominal']}")
    print(f"non-vacuity: component own-source coverage {comp_cov.mean():.4f} (target ~0.90)")
    print(f"wall {res['wall_seconds']}s")


if __name__ == '__main__':
    main()
