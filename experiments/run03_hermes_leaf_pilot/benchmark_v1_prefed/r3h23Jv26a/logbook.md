# logbook.md — reproduction of arXiv 2601.21455 / OpenReview r3h23Jv26a
"Questioning the Coverage-Length Metric in Conformal Prediction: When Shorter Intervals Are Not Better"

Start 2026-08-02 00:56 (+04). End 2026-08-02 02:10. **Elapsed ≈ 1 h 15 min** (soft target 4 h, hard 8 h — well inside).
LLM turns used: ~22 of the 60-turn paper budget. Paper read once → `notes_paper.md`. No official code repo was
found/needed (`notes_code.md` not created — nothing was read from a code repo). No third-party reproduction logs read.

Environment: `.venv` (Python 3.12, numpy/scipy/sympy/pandas/torch-CPU), plus `pip install xlrd` for the `.xls`
concrete file. All runs CPU-only.

---

## Claim 1 — Algorithm 1 (PT) preserves marginal coverage (Theorem 6) while shrinking average length
**Source:** §3.2 Algorithm 1 / Eq (2); Theorem 6 (§3.3); length side: Lemma 1, Theorem 10, Corollary 3 (§3.4).
**Type:** theory + small simulation. **Script:** `verify_claim1.py` → `results/claim1.json`.
**Verdict: `verified`.**

Justifying numbers:
- Symbolic (sympy): `p·(1−α′) = 1−α` with α′ = 1−(1−α)/p — identity holds exactly (`symbolic_identity_holds: true`).
  Hence PT's marginal coverage equals the base's nominal level; over the whole grid α∈{.05,.1,.15,.2,.3} ×
  p∈{.90,…,.99} with p>1−α the maximum |PT coverage − (1−α)| is **0.0** (exact enumeration, 26 cells).
- Finite-sample Monte Carlo (misspecified linear fit on Gaussian-mixture-noise data, **20 seeds**, n=2000/fold):
  | α | p | VCP cov | VCP len | PT cov | PT len |
  |---|---|---|---|---|---|
  |0.10|0.96|0.9018|22.704|**0.9006**|**22.294**|
  |0.10|0.98|0.9018|22.704|0.9015|22.489|
  |0.20|0.96|0.8026|21.782|0.8002|21.169|
  |0.20|0.98|0.8026|21.782|0.8009|21.475|
  PT coverage ≥ nominal − 2·SEM in all four cells; PT length < VCP length in all four cells.

**Mutation test 1 (break the mechanism):** drop the α′ adjustment (use α inside the non-null branch).
Coverage collapses exactly as predicted to ≈ p(1−α): 0.9006 → **0.8663** at α=.1, p=.96 (predicted 0.864),
0.8002 → 0.7706 at α=.2, p=.96 (predicted 0.768). Analytic grid deficit up to **0.085**. Coverage is therefore
carried by the α′ adjustment, not by the randomization per se.

**Mutation test 2 (break the sufficient condition):** replace the misspecified Gaussian-mixture DGP with a
well-specified Gaussian one — paper's failure case, Example 3. PT then gets **longer**: 3.312 → 3.594 (α=.1, p=.96),
2.588 → 2.682 (α=.2, p=.96), while coverage stays at 0.902/0.805. The analytic Example-3 inequality
Φ⁻¹(1−α/2) < p·Φ⁻¹((1+(1−α)/p)/2) was checked at 300+ (α,p) points and holds everywhere.
So the length gain is conditional on the concavity/misspecification condition, exactly as the paper states.

---

## Claim 2 — Table 1: PT-VCP reduces length 22.894 → 22.614 at α=0.10 (p=0.96), coverage preserved
**Source:** Table 1, row α=0.10 / p=0.96; setting App. D.1.1. **Script:** `verify_claim2.py` → `results/claim2.json`.
**Verdict: `verified` (direction and magnitude reproduced) — with one documented specification discrepancy.**

Justifying numbers (20 seeds, sweep of 16 configurations, all with the paper's α=0.10, p=0.96):
- **All 16/16 configurations** give PT shorter than VCP with coverage 0.898–0.903 (nominal 0.90; paper reports
  0.906/0.909).
- Closest configuration (mixture components at ±10, n=1000 per fold): **VCP 22.837 ± 0.058 vs paper 22.894 ± 0.138**
  (Δ = 0.057, inside the paper's own std-err) and **PT-VCP 22.489 ± 0.063 vs paper 22.614 ± 0.254** (Δ = 0.125,
  inside the paper's std-err). Reduction reproduced: −0.35 here vs −0.28 in the paper.
- **Discrepancy:** read literally, App. D.1.1 says ε ~ N(μ,1)/N(−μ,1) with **μ = 20**, i.e. component means ±20.
  That reading gives VCP 43.520 → PT 42.547 — roughly **2× the published scale**. The published 22.894/22.614 are
  recovered only if "μ = 20" denotes the *separation* between the two mixture components (components at ±10).
  β and the fold sizes are not stated in the paper at all; the result is insensitive to β (identical lengths for
  β=(1,1) and (1,2)) and mildly sensitive to n (22.605–23.078 for n=500…5000).

**Mutation:** covered by Claim 1's mutations, which use the same generator (α′ removed ⇒ coverage breaks;
well-specified noise ⇒ PT becomes longer). No additional mutation run for this row.

---

## Claim 3 — Table 2: PT-VCP shortens intervals in 9 of 10 real datasets at 90% coverage
**Source:** Table 2 (§3.4); setting App. D.2.1 / D.2.3 (3-layer 64-64 MLP, Adam 5e-4, bs 64, wd 1e-6, dropout 0.1,
α=0.1, p=0.95, 5 seeds, per-dataset output bias). **Script:** `verify_claim3.py` → `results/claim3.json`
(runtime 1993 s CPU). **Verdict: `inconclusive` for the "9 of 10" statement; partial reproduction on 6 of 10 datasets.**

Datasets attempted (all UCI, downloaded into `./data/`): bike, bio (CASP), concrete, blog-data, facebook-1, facebook-2.
**Not attempted:** meps-19/20/21 (AHRQ MEPS public-use files require registration/data-use agreement plus the CQR
repo's multi-step preprocessing) and star (Harvard Dataverse, gated download). Because 4 of the 10 datasets — and in
particular all three MEPS datasets the anchored claim names — are missing, the literal "9 of 10" count cannot be
confirmed or refuted; it is **not** substituted with a toy.

Result on the 6 attempted datasets (5 seeds, mean ± SEM; paper values in brackets):
| dataset | bias | VCP cov | VCP len | PT cov | PT len | PT shorter? | paper |
|---|---|---|---|---|---|---|---|
| bike | 10 | 0.896 | 20.98±0.03 | 0.896 | **20.24±0.03** | yes | 20.46 → 19.59 (yes) |
| bio | 10 | 0.908 | 21.46±0.02 | 0.903 | **20.66±0.06** | yes | 21.13 → 20.44 (yes) |
| concrete | 5 | 0.888 | 10.55±0.06 | 0.896 | **10.21±0.12** | yes | 10.32 → 9.87 (yes) |
| blog-data | 20 | 0.903 | 43.12±0.46 | 0.898 | **42.72±0.63** | yes | 41.67 → 41.13 (yes) |
| facebook-1 | 10 | 0.895 | 21.98±0.38 | 0.899 | 22.67±0.47 | **no** | 20.81 → 20.80 (tie/yes by 0.01) |
| facebook-2 | 10 | 0.899 | 22.54±0.43 | 0.899 | 23.56±0.68 | **no** | 20.97 → 21.01 (**no** — the paper's 1/10 failure) |

**4 of 6 shorter**, and marginal coverage is within 0.02 of 0.90 on every dataset and both methods
(`all_coverage_within_0.02_of_nominal: true`), so the coverage half of the claim reproduces cleanly.
The two failures land on exactly the datasets the paper itself reports as marginal/negative (facebook-1 is a
0.01 tie in Table 2, facebook-2 is the paper's single negative case), so the qualitative picture agrees; my
facebook-1 goes negative rather than tied, which would make the count 8/10 rather than 9/10 if the pattern held.
Table 3's IS result also reproduces here: IS(PT-VCP) ranges 5.46–95.83 across the six datasets while IS(VCP) = 0.

Deviations from the paper's protocol (CPU budget): ≤8000 rows subsampled per dataset, 40/20/40 train/cal/test split,
≤60 epochs with 10%-validation early stopping (patience 8) instead of cross-validated ≤1000 epochs, features
z-scored on the training fold and the response divided by mean|y_train| (CQR-style). No mutation test is reported
for this claim because it is not marked `verified`.

---

## Claim 4 — Definition 1 + Proposition 2: IS(C_PT) = p(1−p)(E L)² > 0
**Source:** Definition 1 and Proposition 2, §4. **Script:** `verify_claim4.py` → `results/claim4.json`.
**Verdict: `verified`.**

Justifying numbers:
- Symbolic: for the two-point interval-length law {0 w.p. 1−p, L w.p. p}, sympy returns Var = `-L**2*p*(p-1)`
  = p(1−p)L², matching Proposition 2 exactly (`symbolic_matches_prop2: true`).
- Monte-Carlo IS (Definition 1: mean over 500 test points of the variance of |C| over 400 repeated runs, fixed
  calibration set), 5 seeds × p ∈ {0.91,…,0.99} at α=0.10: empirical IS matches the closed form to within
  **2.47 % worst case** across all 35 cells, e.g. p=0.95 → IS 26.27 vs 26.22; p=0.98 → 10.38 vs 10.40;
  p=0.99 → 5.16 vs 5.18. IS is strictly positive everywhere (minimum **4.96**), i.e. the same input really does
  receive different intervals across runs.
- Deterministic VCP on the identical inputs: IS = **3.0e-26** (floating-point zero).

**Mutation test:** set p = 1 (remove the null branch, PT degenerates to its base). Empirical IS drops to
**3.0e-26 ≈ 0**, matching p(1−p)L² = 0. The non-zero stability is therefore produced by the PT randomization
itself, not by the estimator or the calibration noise.

---

## Claim 5 — Proposition 1: PT is a special case of localized CP; IS flags such methods
**Source:** Proposition 1 (§3.2), Definition 1 (§4), Remarks 3 and 12. **Script:** `verify_claim5.py` → `results/claim5.json`.
**Verdict: `verified`.**

Justifying numbers (10 seeds × p ∈ {0.93, 0.95, 0.96, 0.98} × σ̂-floor ε ∈ {1e-2 … 1e-12}, α=0.10):
- With σ̂(x) ∈ {ε, 1} drawn as {1−p, p}, localized CP's normalized-score quantile Q converges to the base
  score quantile at α′ = 1−(1−α)/p: worst-case gap over all seeds/p at ε ≤ 1e-9 is **0.201** on a half-width
  of ≈11.8 (1.7 %), and per-test-point interval lengths differ from the PT reference by at most **0.403** out of
  ≈22 — the entire residual is the finite-sample quantile-index effect (Q is taken as the ⌈(n+1)(1−α)⌉-th smallest
  of n scores of which (1−p)n are +∞, versus the ⌈(m+1)(1−α′)⌉-th of the m = p·n finite ones); it vanishes as n→∞.
  Structurally the intervals are identical: μ̂(x) ± Q·ε ≈ null point-set w.p. 1−p, μ̂(x) ± Q w.p. p — Eq (2).
- This "cherry-picked scale estimator" localized CP is **shorter than VCP on every one of the 40 seed×p cells**
  (`all_localized_shorter_than_vcp: true`), while carrying no task information.
- Interval Stability flags it: IS ≥ **10.08** in every cell (e.g. p=0.93 → 37.30, p=0.98 → 10.48), whereas
  length/coverage alone would rate the method superior.

**Mutation test:** replace the random two-point σ̂ with the deterministic σ̂ ≡ 1. Localized CP then collapses
to VCP exactly — mean length equals the VCP length to < 1e-9 on all 40 cells — and IS falls to **2.8e-26 ≈ 0**.
The flag therefore fires on the randomness, not on the localization.

---

## Final summary table

| Claim | Verdict | One-line evidence |
|---|---|---|
| 1 — Alg. 1 preserves marginal coverage (Thm 6) and shortens length | **verified** | p(1−α′)=1−α exactly (26/26 grid cells, error 0.0); 20-seed MC: coverage 0.9006 vs 0.9018 for VCP, length 22.294 < 22.704; removing the α′ adjustment drops coverage to 0.8663; well-specified Gaussian noise flips length to 3.594 > 3.312 (Example 3). |
| 2 — Table 1: 22.894 → 22.614 at α=0.10, p=0.96 | **verified** (spec discrepancy noted) | Best config reproduces 22.837 → 22.489 (both inside the paper's std-errs), 16/16 configs shorter; but the literal "μ=20" reading gives 43.520 → 42.547, i.e. 2× the published scale. |
| 3 — Table 2: 9 of 10 real datasets shorter at 90% coverage | **inconclusive** | 6/10 datasets obtainable (MEPS ×3 need AHRQ approval, STAR is Dataverse-gated); on those, 4/6 shorter with coverage 0.888–0.908, and the two failures are the two facebook variants the paper itself reports as tie/negative. |
| 4 — Prop. 2: IS(C_PT) = p(1−p)(E L)² > 0 | **verified** | sympy gives exactly p(1−p)L²; empirical IS matches within 2.47 % over 35 cells, min IS 4.96 > 0; VCP IS = 3e-26; mutation p=1 ⇒ IS = 3e-26. |
| 5 — Prop. 1: PT ⊂ localized CP, and IS detects it | **verified** | Two-point σ̂ localized CP matches PT intervals to ≤0.403/22 (finite-sample index only), is shorter than VCP in 40/40 cells, and IS ≥ 10.08; with σ̂ ≡ 1 it becomes VCP exactly and IS = 3e-26. |

Gate: `.venv/bin/python check_reproducibility.py` re-asserts every number above against `results/*.json`.

---

## Evidence boundary — what this evidence does NOT cover

1. **Claim 3 is only partially covered.** MEPS-19/20/21 and STAR were never downloaded (registration / gated
   Dataverse), so the headline "9 of 10" count is untested. Facebook-1 came out negative here whereas the paper
   reports it as a 0.01 tie; with only 6 datasets I cannot say whether the paper's count is 9/10, 8/10 or exact.
2. **Claim 3's training protocol is not the paper's.** ≤8000-row subsamples, ≤60 epochs with simple early stopping
   (paper: cross-validated schedule up to 1000 epochs), a 40/20/40 split the paper does not specify, and my own
   feature/response scaling. Absolute lengths therefore differ from Table 2 by up to ~1.6 (blog-data) and the
   sign of small per-dataset differences (facebook) is not robust at this budget.
3. **Claim 2's absolute numbers rest on a re-interpretation** of "μ = 20" and on guessed values for β and the
   fold sizes, none of which the paper states. Under the literal reading the Table 1 numbers do not reproduce.
   I did not contact the authors or locate an official code release to settle this.
4. **No official code was consulted.** No repository link was found in the rendered paper text; everything here is
   a from-scratch re-implementation of Algorithm 1 / Algorithm 2 as described. Implementation-level differences
   (quantile-index convention, null set = point vs empty set) could shift small numbers.
5. **Only the absolute-residual VCP base was tested.** CQR (Table 5), the classification/RAPS experiments
   (Table 4), and the group-coverage results (Figure 1, Table 6) were not attempted at all.
6. **Theorems 7, 9, 10, Corollaries 1–3 and Lemma 1 were not independently proof-checked**; Theorem 6 and
   Proposition 2 were verified symbolically plus empirically, Proposition 1 empirically (finite-n, one DGP),
   and Example 3 only through its final inequality (Eq 54) evaluated numerically, not through its proof.
7. **All simulations use one synthetic DGP family** (2-dim Gaussian X, linear β, Gaussian-mixture noise) and a
   single score function. The coverage results are distribution-free and should transfer; the *length* results
   are explicitly condition-dependent (Corollary 3 / Example 3) and do not generalize beyond the settings tested.
8. **Interval Stability was measured with a fixed calibration set** (as Definition 1 prescribes: Var conditional
   on X and D_ca). Variability across calibration draws — the thing practitioners often care about — is not measured.
