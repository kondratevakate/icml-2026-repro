## Evidence boundary — what this evidence does **not** cover

1. **Claims 1–3 are verified on discrete, finite `(X, Y)` worlds with counting measure μ.** Continuous-Y
   regression settings (Lebesgue μ, density estimation, interval-valued sets) are **not** tested; the
   paper's regression instantiation (Sec. 4.3) is untouched by my scripts.
2. **The paper's actual estimators are not tested.** Claims 1–3 use my own faithful re-implementation of
   the *mathematical* objects (max-p p-values, LP duals, plug-in `f̂`, LP-based `λ̂`). The authors'
   `model/MDCP.py` (LambdaSpline, γ-penalty tuning, GBM/Gaussian source models) was **read but never
   executed to completion**, so nothing here validates their code.
3. **Claim 2's "exact 1 − α for at least one source" is verified conditionally on x, with μ finite.**
   Uniqueness under `μ({h_{λ*}=1}) = 0` is not separately tested (that null-tie regime is empty in my
   discrete instances — indeed ρ(T) > 0 there).
4. **Claim 3 is an empirical convergence study, not a proof.** Largest n is 25 600 per source; rates are
   not estimated, only the trend and the `ρ(T)` bound. Sup-norm consistency of `f̂`, `λ̂` was assumed by
   construction rather than verified under nonparametric conditions.
5. **No efficiency claim is reproduced at all.** Every quantitative efficiency number in the paper
   (34.39%, ~22%, 26.3% median, Fig. 2/5/8) remains unchecked; my claim-1 experiment even shows a
   *conservative* max-p set (19.2/20 labels), which is consistent with the paper's premise but says
   nothing about how much MDCP shrinks it.
6. **No real data was used.** FMoW, PovertyMap and MEPS results, the six-region worst-case coverage, and
   the 249-country statement about the training data are entirely unverified here.
7. **Monte-Carlo resolution.** Claim-1 coverages rest on ~1333 test points per source per world
   (99% binomial band ±~2 points at 0.9); claim-3 coverages on 6000 draws per cell. Deviations below
   that resolution would not have been detected.
8. **Single machine, single software stack**, no seed-sensitivity study beyond the 8 worlds / 30 seeds
   per configuration reported above.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Evidence boundary \u2014 what this evidence does **not** cover"}\n-->
