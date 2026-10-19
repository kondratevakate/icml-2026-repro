"""ProConMV (orid 4F4Ziv1d9G) Corollary 1 core check (groundwork).
Corollary 1: dynamic fusion weights negatively correlated with per-view losses lower the
generalization bound vs static uniform weights. We verify the dominant data-dependent term
(the Term-L average empirical loss of Theorem 1) via the covariance identity
E[w L] = E[w]E[L] + Cov(w,L): Cov(w,L)<0 => weighted loss below uniform. Rademacher/concentration
terms of the full bound are not computed. numpy, CPU, seconds.
Note: ProConMV's two CHALLENGE claims are empirical (MFIDDR/DRTiD + GPU); this only verifies the
supporting theorem behind the dual-uncertainty module."""
import numpy as np
rng = np.random.default_rng(0)
V, trials = 4, 20000
lower = 0; gaps = []
for _ in range(trials):
    L = rng.uniform(0.1, 2.0, V)
    u = L * rng.uniform(0.7, 1.3, V)         # Dirichlet uncertainty grows with loss (Sensoy 2018)
    wd = 1.0 / u; wd /= wd.sum()             # dynamic: inverse uncertainty
    ws = np.ones(V) / V                      # static uniform
    dyn = float((wd * L).sum()); sta = float((ws * L).sum())
    gaps.append(sta - dyn); lower += dyn < sta
gaps = np.array(gaps)
print(f"dynamic < static in {lower/trials*100:.1f}% of {trials} trials; "
      f"mean advantage {gaps.mean():.4f}+-{gaps.std():.4f}")
