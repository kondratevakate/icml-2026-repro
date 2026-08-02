# Plan — reproduction of "Multi-Distribution Robust Conformal Prediction" (arXiv 2601.02998, ICML 2026)

Paper read: arXiv HTML v2 (`paper/paper.html`, text dump `paper/paper.txt`).
Official code read (read-only): `code_mdcp/` (`model/MDCP.py`, `notebook/eval_linear.py`, `notebook/eval_utils.py`, README).
No third-party reproduction logbooks were consulted.

| # | Anchored claim | Type | CPU-feasible? | Approach |
|---|---|---|---|---|
| 1 | Thm 1 (Sec. 2): finite-sample uniform validity of max-p aggregation; set = union of per-source sets | theory + exact simulation | yes | Discrete (X,Y) world where the per-source laws are known exactly; max-p with the finite-sample `(1+#{>=})/(n+1)` p-value; exhaustive over sources (= simplex vertices; mixture coverage is the convex combination of them) plus an 11x11 mixture grid; 8 independent worlds x 4000 draws. Mutations: drop the `+1` correction, mean-p instead of max-p, single-source calibration. |
| 2 | Thm 2 (Sec. 3.1): optimal set = superlevel set of `h_{λ*}(x,y)=Σ_k λ_k*(x) f_k(y|x)`; complementary slackness gives exact 1−α for ≥1 source | analytic / LP | yes | With μ = counting measure on a finite Y-grid the conditional program (4) is an LP; solve with `scipy.optimize.linprog` (HiGHS), read exact duals λ*, rebuild `{h_{λ*}>1} ∪ S`, compare objective and coverage. Exhaustive over K∈{2..5} × m∈{3..10} × α∈{.05,.1,.2} × 30 seeds = 2880 instances. Mutations: uniform λ, inverted superlevel condition, drop an active constraint. |
| 3 | Thm 3 (Sec. 3.2): max-p with learned scores is asymptotically optimal up to the boundary set T | simulation with growing n | yes | Exactly the paper's estimator: `ĥ = Σ_k λ̂_k(x) f̂_k(y|x)`, score `s_k=-ĥ`, randomized p-values, `Ĉ={y : max_k p^{(k)} ≥ α}`; f̂ from multinomial counts, λ̂ by re-solving the LP on f̂; n ∈ {100,…,25600}, 8 worlds, 120 randomizations. Checks the exact inequality `‖Ĉ|−|C*‖ ≤ ρ(T)` and the "ρ(T)=0 ⇒ symmetric difference → 0" corollary (measured outside T). Mutations: frozen uniform λ̂; single-source density score. |
| 4 | Linear classification: MDCP sets 34.39% smaller than Baseline-agg (Fig. 2) | empirical, official code | attempted | Run the authors' `notebook/eval_linear.py` unmodified on CPU (paper averages **100 trials**); compute (size(Baseline-agg) − size(MDCP))/size(Baseline-agg). |
| 5 | Linear regression: 22.44% average reduction, ~90% worst-case coverage (Fig. 5) | empirical, official code | attempted | Same run, regression payload. Paper prose says "about 22% narrower". |
| 6 | FMoW (six regions, 249 countries), near-tight worst-case coverage (Sec. 6.1, Fig. 8) | empirical, needs data + GPU training | **no — refuse** | Requires the WILDS FMoW v1.1 download (>100 GB, 1M+ images) and DenseNet-121 training for 30 epochs over 100 splits. Not attainable on CPU within a ~75-minute budget; recorded as `inconclusive` with the reason rather than substituted by a toy. |

Budget policy: claims 1–3 get full treatment (exhaustive enumeration + mutation tests);
claims 4–5 are run with the authors' own pipeline for as many trials as the budget allows and are
reported honestly as under-powered if fewer trials than the paper's 100 complete; claim 6 is refused.
