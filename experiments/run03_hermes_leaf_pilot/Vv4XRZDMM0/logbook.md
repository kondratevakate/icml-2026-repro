# Reproduction logbook — *Multi-Distribution Robust Conformal Prediction* (Yang & Jin, ICML 2026)

- Paper: arXiv:2601.02998v2 (HTML read into `paper/paper.txt`), OpenReview `Vv4XRZDMM0`
- Official code (read-only): `code_mdcp/`, working copy `mdcp_run/`
- Hardware/env: CPU only, WSL2, `.venv` = python 3.12 / numpy 2.5.1 / scipy / torch 2.13.0+cpu / sklearn 1.9.0
- Reproducibility gate: `python check_reproducibility.py` → 54 quoted values, exit 0
- No third-party reproduction logbook or HF Space was read.

| # | Claim (source location) | Verdict | Evidence in one line |
|---|---|---|---|
| 1 | Thm 1, Sec. 2 — finite-sample uniform validity of max-p | **verified** | worst-source coverage ≥ 0.9767 in 8/8 worlds (0 below 0.9), union identity exact in 32 000 draws; single-source calibration collapses to 0.5514 |
| 2 | Thm 2, Sec. 3.1 — superlevel set of `h_{λ*}` + complementary slackness | **verified** | 2880/2880 LP instances: superlevel size = LP optimum (max gap 3.1e-14), CS (i)(ii)(iii) all hold; every mutation breaks feasibility/optimality |
| 3 | Thm 3, Sec. 3.2 — asymptotic optimality up to boundary `T` | **verified** | symmetric difference outside `T` falls 1.2823 → 0.0546 as n: 100 → 25 600 (8 worlds); `‖Ĉ|−|C*‖ ≤ ρ(T)` in 40/40 world×n cells; inconsistent-λ and single-density mutations plateau at 0.3577 / 1.6233 |
| 4 | Linear classification, 34.39% smaller than Baseline-agg (Sec. 5.2, Fig. 2) | **inconclusive** | authors' `eval_linear.py` ran 1780 s CPU and completed 0 of the 100 trials the paper averages — no number produced, none invented |
| 5 | Linear regression, 22.44% reduction, ~90% worst-case coverage (Sec. 5.3, Fig. 5) | **inconclusive** | same run; regression stage never reached. Paper prose says "about 22% narrower" (the 22.44% digit lives only in the figure) |
| 6 | FMoW, six regions / 249 countries (Sec. 6.1, Fig. 8) | **inconclusive** | needs WILDS FMoW v1.1 (>100 GB, >1M images) + DenseNet-121 × 30 epochs × 100 splits; infeasible CPU-only in budget, and no toy substitute was made |

---

## Claim 1 — Theorem 1 (Section 2), finite-sample uniform validity

**Statement tested.** `Ĉ(X_{n+1}) = ∪_k Ĉ^{(k)}(X_{n+1})`, and for a test point from **any**
mixture `P = Σ_k π_k P^{(k)}`, `P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 − α`, for arbitrary conformity scores.

**Script.** `verify_claim1.py` → `results/claim1.json` (runtime 170.8 s).
Discrete world, K = 3 sources, |X| = 4, |Y| = 20, n_cal = 50 per source, α = 0.1,
8 independent worlds × 4000 draws each, p-values `(1 + #{S_i ≥ s})/(n+1)`.

**Numbers.**
- worst-per-source coverage: **min 0.9767**, mean 0.9861 over 8 worlds; **0/8 worlds below 1 − α = 0.9**.
- Exhaustive over the mixture simplex: coverage under a mixture is the π-weighted average of the
  per-source coverages, so the minimum is attained at a vertex; checked directly on **968** weight
  vectors (11×11 grid × 8 worlds) → **worst mixture coverage 0.9767**.
- Union identity `Ĉ = ∪_k Ĉ^{(k)}`: **0 mismatches** out of 32 000 test points.
- Mean set size 19.20 / 20 labels → the aggregated set is *valid but very conservative*, exactly the
  "Baseline-agg is conservative" behaviour the paper motivates its method with.

**Mutation tests.** (all: same worlds/seeds, one mechanism changed)
| mutation | worst-source coverage (min / mean) | predicted | observed |
|---|---|---|---|
| drop the finite-sample `+1` correction | 0.9662 / 0.9801 | strictly lower coverage | ✔ lower everywhere, still above 0.9 at this conservativeness level |
| mean-p instead of max-p | 0.9520 / 0.9641 | loses the guarantee's slack | ✔ drops by 3.4 points, smaller sets (18.15) |
| **single-source calibration** (use only source 0's p-value) | **0.5514 / 0.6460** | severe under-coverage on the other sources | ✔ **8/8 worlds below 0.9** — the max over sources is the mechanism, not the score quality |

**Verdict: verified.** Coverage never fell below 1 − α anywhere in the design; deleting the max-over-sources
step destroys validity by ~35 coverage points.

---

## Claim 2 — Theorem 2 (Section 3.1), X-conditional optimality

**Statement tested.** For fixed x the program (4) has optimum
`C*(x) = {y : h_{λ*}(x,y) > 1} ∪ S(x)`, `h_λ = Σ_k λ_k(x) f_k(y|x)`, with complementary slackness
(i) `λ_k*>0 ⇒` exact 1 − α, (ii) `λ_k*=0 ⇒ ≥ 1 − α`, (iii) some `λ_{k*}* > 0`.

**Script.** `verify_claim2.py` → `results/claim2.json` (runtime ≈ 81 s). With μ = counting measure on a
finite Y-grid, (4) is an LP; solved with `scipy.optimize.linprog` (HiGHS) and its exact duals.
**Exhaustive enumeration of the design axes**: K ∈ {2,3,4,5} × m ∈ {3,…,10} × α ∈ {0.05, 0.1, 0.2}
× 30 random density draws = **2880 instances**.

**Numbers.**
- Superlevel-set solution reproduces the LP optimum in **2880/2880** instances, max size gap **3.1e-14**.
- Strong duality gap (primal − dual (5)) ≤ **4.0e-14**.
- Complementary slackness (i)/(ii)/(iii): **2880 / 2880 / 2880**.
- "Exact 1 − α coverage for at least one source": **2880/2880**.

**Mutation tests.**
| mutation | result |
|---|---|
| λ* → uniform λ with the same ℓ1 mass | superlevel set infeasible in **2680/2880**, infeasible *or* strictly larger in **2880/2880** |
| invert the superlevel condition (`{h_{λ*} < 1}`) | infeasible in **2880/2880** |
| drop the largest active constraint | LP optimum strictly shrinks in **2847/2880** (the remaining 33 are ties where another source binds identically) — i.e. active constraints really are the binding ones |

**Verdict: verified** (to solver precision, on the discrete-μ instantiation of the theorem).

---

## Claim 3 — Theorem 3 (Section 3.2), asymptotic optimality

**Statement tested.** With `ĥ = Σ_k λ̂_k(x) f̂_k(y|x)`, score `s_k = −ĥ`, the paper's randomized
p-values and `Ĉ^{(n)}(x) = {y : max_k p^{(k)} ≥ α}`:
`limsup_n ||Ĉ^{(n)}| − |C*|| ≤ ρ(T)`, `T = {(x,y) : h*(x,y) = 1}`, and if ρ(T)=0 then
`ρ(Ĉ^{(n)} △ C*) → 0`.

**Script.** `verify_claim3.py` → `results/claim3.json` (runtime 21.3 s). K = 3, |X| = 5, |Y| = 12,
α = 0.1; oracle λ*(x), h*(x,·), `{h*>1}` and `T(x)` computed exactly per x by LP; `f̂_k` from
multinomial training counts and `λ̂(x)` by re-solving the LP on `f̂` (both consistent, as Thm 3 assumes);
n per source ∈ {100, 400, 1600, 6400, 25600}, 8 worlds, 120 randomizations each.

**Numbers.**
- Boundary mass in these worlds is **not** negligible: mean `ρ(T) = 2.875` labels per x — the LP optimum is
  fractional on `T`, so the theorem's slack term is genuinely active. The theorem's inequality
  `||Ĉ| − |C*|| ≤ ρ(T)` held in **8/8 worlds at all 5 sample sizes (40/40 cells)**.
- Symmetric difference to `{h*>1}` **restricted to the complement of T** (the part Thm 3 controls):
  **1.2823 → 0.4696 → 0.4654 → 0.0694 → 0.0546** for n = 100 … 25 600. Converging to 0 as claimed.
- Finite-sample validity survives throughout: min worst-source coverage ≥ **0.874** across all cells
  (Theorem 1 does not depend on the score, as the paper stresses).

**Mutation tests** (same worlds, only the score-learning mechanism broken):
| mutation | symmetric difference outside T at n = 25 600 | predicted |
|---|---|---|
| main (consistent λ̂, f̂) | **0.0546** | → 0 ✔ |
| λ̂ frozen at a uniform vector (violates `sup_x‖λ̂−λ*‖→0`) | **0.3577** (flat from n = 1600 on) | does not vanish ✔ |
| score built from a single source density `f̂_0` | **1.6233** (flat) | does not vanish ✔ |

**Verdict: verified** — convergence happens only when the premises of Thm 3 hold, and stalls at a
positive plateau when either premise is broken, while coverage stays valid in every arm.

---

## Claim 4 — linear classification, "34.39% smaller than Baseline-agg" (Sec. 5.2, Fig. 2)

Wording confirmed verbatim in the paper text: *"MDCP sets are on average 34.39% smaller than Baseline-agg."*

**What was attempted.** The authors' own pipeline, unmodified:
`python notebook/eval_linear.py --num-trials 3 --base-seed 34567 --seeds 34567 34568 34569`
(`mdcp_run/` = copy of `code_mdcp/`, `.venv` interpreter, `matplotlib` added only because
`notebook/simulation.py` imports it at module load).

**What happened.** After **1780 s** of CPU the run was still inside trial 1: it had fitted the three
source models and the pooled GBM and entered the `GAMMA_GRID = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]`
loop (`eval_linear.py:315`), which fits the λ-spline and evaluates MDCP on 3000 test points for each γ,
twice per trial (classification + regression). **0 of 3 trials finished**; the paper averages **100** trials.
Log: `results/eval_linear_official_run.log`; machine-readable status: `results/claim4_5.json`.

**Verdict: inconclusive** — no size-reduction number was measured, so none is reported. Nothing here
supports or contradicts 34.39%; the honest statement is that the official pipeline needs far more than a
75-minute CPU budget for even one of the 100 trials.

## Claim 5 — linear regression, "22.44% reduction, ~90% worst-case coverage" (Sec. 5.3, Fig. 5)

Same run; the regression stage was never reached. Note also a **provenance discrepancy worth flagging**:
the paper's *text* says "about 22% narrower than Baseline-agg in Linear", and the appendix reports a
median **26.3%** reduction across the 120 regression configurations; the exact figure **22.44%** appears
only in the Figure-5 panel, i.e. it is not checkable from the paper text alone.

**Verdict: inconclusive** (budget, not evidence against).

## Claim 6 — FMoW, six regions, 249 countries (Sec. 6.1, Fig. 8)

Reproduction path per README: download WILDS FMoW v1.1 (>100 GB, >1M images), create 2016 region splits,
train DenseNet-121 for 30 epochs, predict, then learn λ / calibrate over 100 splits. Not attainable
CPU-only inside the budget; the download was not started. Only the paper-text side facts were confirmed
(six regions Africa/Americas/Asia/Europe/Oceania/Other; "249 countries/regions" appears in Sec. 6.1).

**Verdict: inconclusive.** A synthetic stand-in was deliberately *not* built.

---

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
