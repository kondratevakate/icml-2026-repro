# Reproduction logbook — Semi-knockoffs (ICML 2026)

**Paper.** Angel Reyero Lobo, Bertrand Thirion, Pierre Neuvial, *Semi-knockoffs: a model-agnostic
conditional independence testing method with finite-sample guarantees*, ICML 2026.
OpenReview `Xf9hJMGwDd` · arXiv `2601.23124` (read: **v2**, 20 Jun 2026, HTML).
Official code: `https://github.com/AngelReyero/loss_based_KO` → cloned to `./official_code`
(`src/semi_KO.py`, `src/utils.py::knockoff_threshold` were read and followed).

**Environment.** WSL2 Ubuntu, Python 3.12.3, venv at `./.venv`, numpy 2.5.1 / scipy 1.18.0 /
sympy 1.14.0. CPU only. **No scikit-learn** (task dependency restriction) — this is what caps
claims 5 and 6.

**No third-party reproduction logbooks were consulted.** Sources used: the arXiv paper and the
authors' own repository only.

---

## Summary table

| # | Claim (short) | Source | Verdict |
|---|---|---|---|
| 1 | Paired nonparametric test gives valid p-values with **no train-test split**, needing only ν_j, ρ_j | Thm 3.3, §3.1 | **verified** (oracle hypothesis) — with a documented caveat on the plug-in version |
| 2 | FDR(S_SKO) ≤ q | Thm 3.4, §3.2 | **verified** |
| 3 | ‖θ̃^j − θ̂‖₂ ≤ O_P(√(log(1/δ)/n)) for null j | Thm 4.1, Eq. (3), §4.2 | **verified** |
| 4 | Double robustness, compound rate O_P(a_n b_n) | Thm 4.3, §4.4 | **inconclusive** (not run — see below) |
| 5 | SKO controls type-I and beats HRT in power; derandomization (5 perms) raises power | Figs 4/5, §5.1; App F.4.2 | **toy** (directionally supported, substitute black box) |
| 6 | WDBC real data across RF / NN / GB | Fig 6, §5.2; App F.6 | **inconclusive** (no sklearn, no real-data pipeline) |

---

## Claim 1 — Theorem 3.3 (Type-I error), §3.1

> *"Given ν_j and ρ_j, nonparametric paired test Semi-knockoffs (Algorithm 2) provide valid p-values."*

**Operationalisation.** A p-value is valid iff super-uniform under H₀: P(p ≤ α) ≤ α. Estimated over
independent seeds on the paper's adjacent-support design (X ~ N(0, Σ), Σ_ij = 0.6^|i−j|,
β[:0.25p] ~ U[1,2], y = β'X + ε), testing coordinates with β_j = 0. Crucially the black box (ridge)
is fitted on **all n samples with no split** — the regime the theorem claims to legitimise.

**Scripts.** `verify_claim1.py` (600 seeds, n=200, p=10), `verify_claim1b.py` (7 (n,p) configs × 400 seeds).

### Arm A — oracle ν_j, ρ_j (the theorem's actual hypothesis)

`results/claim1.json`, n=200, p=10, 600 seeds, MC s.e. ≈ 0.009/0.012/0.016:

| α | sign test | Wilcoxon |
|---|---|---|
| 0.05 | **0.0367** | 0.0550 |
| 0.10 | **0.0767** | 0.1017 |
| 0.20 | **0.1850** | 0.1917 |

`results/claim1b.json`, sweep at α = 0.10 (MC s.e. 0.015), rejection rate of the oracle arm:

| n | 100 | 200 | 400 | 800 | 1600 | 100 (p=40) | 200 (p=40) |
|---|---|---|---|---|---|---|---|
| oracle | 0.0850 | 0.0700 | 0.0825 | 0.0775 | 0.1000 | 0.0800 | 0.0675 |

Super-uniform at **every** configuration tested, both tests, all α. ✔

### Mutation test

Replace the symmetric two-sided construction (Algorithm 1) with the HRT-style **asymmetric**
statistic `l(m̂(X̃₁^(j)), y) − l(m̂(X), y)` evaluated on the training data — i.e. exactly the
construction §3 says requires "independence between the test set and the model". Prediction:
validity is destroyed. Result (`C_mutation_asymmetric_HRT_style_no_split`, 600 seeds, sign test):

| α | oracle SKO | mutation | ratio |
|---|---|---|---|
| 0.05 | 0.0367 | **0.0867** | 2.4× |
| 0.10 | 0.0767 | **0.1683** | 2.2× |
| 0.20 | 0.1850 | **0.3133** | 1.7× |

And in the sweep the mutation sits at 0.14–0.20 (vs nominal 0.10) at every n up to 1600, with
**no decay** — the violation is structural, not a small-sample artefact. The symmetric pairing is
therefore the mechanism, not a correlate. ✔

**Verdict: `verified`** for Theorem 3.3 as stated ("Given ν_j and ρ_j").

### Caveat that my evidence also produced (arm B — the *practical* Algorithm 1)

When ν̂_j and ρ̂_j are ridge-estimated from the same data (Algorithm 1, §4.1) instead of oracle,
the type-I error was **not** controlled in my runs, and did not improve with n:

| n (p=10) | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|
| estimated ν̂, ρ̂ (α=0.10) | 0.178 | 0.188 | 0.138 | 0.215 | 0.173 |

This does **not** contradict Theorem 3.3 (which conditions on the oracle) nor Theorem 4.2 (which
gives 𝒲₁(P̂₁,P̂₂) = O_P(n^{-1/2}), a statement about the *distributions*, not the test). A plausible
reading: the discrepancy shrinks at n^{-1/2} while the paired test's ability to detect it grows at
n^{1/2}, so the rejection rate does not converge to α. I flag this as an open discrepancy between
the theorem's guarantee and the deployed algorithm in my configuration, not as a falsification —
my imputer (plain ridge, λ=1e-2) is not necessarily the authors' choice.

---

## Claim 2 — Theorem 3.4 (FDR control), §3.2

> *"Given ν_j and ρ_j for every j, then FDR(S_SKO) ≤ q."*

**Script.** `verify_claim2.py`. n=300, p=20, 5 non-null (adjacent block), n_perm=5,
**oracle** ν_j, ρ_j, ridge black box fitted on all n samples, 300 seeds per configuration.
W_SKO^j = mean loss under X̃₁ − mean loss under X̃₂; threshold = Eq. (1) exactly as in the paper
and in `official_code/src/utils.py::knockoff_threshold`; FDP := 0 when nothing is selected.

`results/claim2.json`:

| q | **FDR (SKO)** | MC s.e. | power | mean #selected |
|---|---|---|---|---|
| 0.05 | **0.000** | 0.000 | 0.000 | 0.00 |
| 0.10 | **0.024** | 0.007 | 0.043 | 0.49 |
| 0.20 | **0.156** | 0.011 | **1.000** | 6.37 |

FDR ≤ q at every level. The q=0.20 row is the non-vacuous one: full power (all 5 true signals
recovered every time) while FDR stays at 0.156 < 0.20, i.e. control is not achieved by refusing
to select. ✔

### Mutation tests

**M1 — remove the `1 +` offset from Eq. (1)** (the finite-sample correction the martingale argument
depends on). Prediction: FDR > q.

| q | SKO (offset=1) | **M1 (offset=0)** | violates q? |
|---|---|---|---|
| 0.05 | 0.000 | **0.135** | yes (2.7×) |
| 0.10 | 0.024 | **0.155** | yes (1.6×) |
| 0.20 | 0.156 | **0.269** | yes (1.3×) |

**M2 — break sign exchangeability (Lemma 2.1)** by using the asymmetric no-split statistic.
Prediction: FDR ≫ q.

| q | SKO | **M2** |
|---|---|---|
| 0.05 | 0.000 | **0.078** |
| 0.10 | 0.024 | **0.675** |
| 0.20 | 0.156 | **0.712** |

At q=0.10 the mutation selects 16.0 of 20 features with 67.5 % of them false. Catastrophic, as predicted.

**M3 — negative control, ρ_j := ν_j on both sides.** Signs stay symmetric, so FDR should remain
controlled, but the alternative becomes undetectable. Result: FDR = 0.000 / 0.000 / 0.010 and
power = 0.000 / 0.000 / 0.031. Exactly as predicted — this isolates which ingredient buys *control*
(the symmetry) from which buys *power* (conditioning ρ on y). ✔

**Verdict: `verified`.**

---

## Claim 3 — Theorem 4.1 (Optimization stability), Eq. (3), §4.2

> *"For j ∈ H₀ … ‖θ̃^j − θ̂‖₂ ≤ O_P(√(log(1/δ)/n)) with probability greater than 1−δ."*

**Script.** `verify_claim3.py`. Quadratic loss — one of the two losses the paper says it verifies
the regularity assumptions for — so the regularised ERM of Eq. (2) has the **closed form**
θ̂ = (χ'χ/n + λI)^{-1} χ'z/n and the measured quantity is exact (no optimiser noise).
DGP is Figure 1 verbatim: p=50, β 0.25-sparse with important features in blocks of 5,
χ ~ N(0, 0.6^|i−j|), noise ‖χβ‖/2, λ=1e-2. 250 replicates per n, n ∈ {125…4000}.

The bound has **two** functional contents, tested separately.

**T1 — rate in n.** 0.9-quantile of ‖θ̃^j − θ̂‖₂ for null j:

| n | 125 | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|
| q₀.₉ | 1.428 | 0.700 | 0.532 | 0.349 | 0.246 | 0.168 |

log-log slope = **−0.587** (R² = 0.983) vs predicted −0.5. Since Eq. (3) is an **upper** bound,
decay at least as fast as n^{-1/2} is what it asserts; observed decay is slightly faster. ✔

**T2 — rate in δ.** At fixed n, regressing quantile(1−δ) on √(log(1/δ)) through the origin over
δ ∈ {0.5, 0.2, 0.1, 0.05, 0.02, 0.01}:

| n | slope | R² |
|---|---|---|
| 500 | 0.365 | **0.938** |
| 2000 | 0.162 | **0.953** |
| 4000 | 0.120 | **0.911** |

Near-linear in √(log(1/δ)) at every n, and the slope itself shrinks with n roughly as n^{-1/2}
(0.365 → 0.162 → 0.120 for n ×4, ×2), i.e. the two dependencies factorise as the bound claims. ✔

### Mutation test

Run the identical estimator on a **non-null** coordinate (β_j ≠ 0), violating the theorem's
`j ∈ H₀` hypothesis. Prediction: no n^{-1/2} decay, plateau near |β_j|.

| n | 125 | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|
| q₀.₉ (non-null) | 3.222 | 2.742 | 2.391 | 2.324 | 2.270 | 2.268 |

log-log slope = **−0.097** (vs −0.587 for null), flat from n=1000 onward. At n=4000 the non-null
quantity is **13.5×** the null one, and the gap widens with n. The decay is therefore driven by the
nullity of the feature, not by ridge shrinkage or by generic convergence of θ̂. ✔

**Verdict: `verified`.**

---

## Claim 4 — Theorem 4.3 (double robustness), §4.4

**Verdict: `inconclusive`. Not executed.** See `results/claim4.json`.

Reason: distinguishing the compound rate O_P(a_n b_n) from the additive alternative O_P(a_n + b_n)
requires a two-dimensional grid with the model error a_n and the sampler error b_n tuned
*independently*; a one-axis experiment cannot separate them and would be a fake verification.
That did not fit the budget at the standard applied to claims 1–3. Note also that the paper itself
presents this result as supporting a **conjecture** of control (§1: "conjecturing the control";
§4.5: "should preserve the sign … with high probability"), so a half-run number here would be
unusually easy to over-read. What would be needed is documented in `results/claim4.json`.

---

## Claim 5 — Figures 4/5, §5.1 (+ masked correlation, App F.4.2)

**Verdict: `toy`** — and the cap is structural, not a hedge: the paper's black box in these figures
is a neural network / gradient boosting / random forest, and with sklearn unavailable mine is a
**ridge regression**. This is not the paper's configuration.

**First attempt (`verify_claim5.py`, `results/claim5.json`)** was uninformative: at n=200 with a
strong linear signal, SKO-1, SKO-5 and HRT all reached **power = 1.000**. A ceiling effect — the
anchored comparison is untestable there. Recorded rather than discarded.

**Second attempt (`verify_claim5b.py`, `results/claim5b.json`)**, harder regime (oracle ν/ρ, so
that the split effect is isolated from imputer error), α=0.05, 250 seeds, K_HRT=100:

| config | method | type-I | power |
|---|---|---|---|
| n=60, p=12, σ=6 | SKO-1 | 0.042 | 0.125 |
| | **SKO-5** | 0.045 | **0.331** |
| | HRT | 0.060 | 0.145 |
| n=80, p=12, σ=4 | SKO-1 | 0.044 | 0.256 |
| | **SKO-5** | 0.045 | **0.725** |
| | HRT | 0.048 | 0.373 |
| n=120, p=12, σ=4 | SKO-1 | 0.042 | 0.295 |
| | **SKO-5** | 0.038 | **0.845** |
| | HRT | 0.048 | 0.527 |

Findings, stated with their limits:
- **Type-I control (SKO):** holds at ≤ 0.045 in all three configs, below the nominal 0.05. Supports
  the first half of the claim.
- **Derandomization with 5 permutations increases power:** strongly supported — 0.125→0.331,
  0.256→0.725, 0.295→0.845, all far beyond MC s.e. (~0.016), with type-I unchanged.
- **"SKO has higher power than HRT":** supported **only for the derandomized SKO-5**
  (0.331 > 0.145, 0.725 > 0.373, 0.845 > 0.527). The single-permutation SKO-1 was consistently
  *weaker* than HRT (0.125 < 0.145, 0.256 < 0.373, 0.295 < 0.527) in my configuration. The paper's
  Figures 4/5 present both variants; my evidence reproduces the advantage for the derandomized one
  and does not reproduce it for n_perm=1.
- In the **masked-correlation** setting with *plug-in* (empirical) conditional means
  (`results/claim5.json`), SKO type-I rose to 0.092 (SKO-1) and 0.250 (SKO-5) — consistent with the
  claim-1 arm-B caveat: with estimated ν̂/ρ̂, derandomization amplified type-I rather than power.

No mutation test is reported for claim 5, and consequently it is **not** marked verified.

---

## Claim 6 — Figure 6, §5.2 (WDBC real data, RF / NN / GB)

**Verdict: `inconclusive`.** See `results/claim6.json`.

The claim is specifically about a *real* dataset and *three specific* model families. scikit-learn
— which supplies both `load_breast_cancer` and all three estimators, and on which the authors'
`src/experiments/real_data.py` depends — is not available under this task's dependency restriction.
Substituting a synthetic dataset or a linear model would be a toy standing in for real data, which
the task rules forbid. Not attempted.

---

## Evidence boundary

What the evidence above does **NOT** cover:

1. **Black-box models.** Every executed number uses a **ridge / linear** black box m̂. Nothing here
   speaks to the paper's neural network, gradient boosting or random forest results, which are the
   configurations in Figures 4, 5, 6 and Appendix F. The model-agnosticism claim is therefore
   **untested** — I tested one model class, not agnosticism.
2. **Real data.** No real dataset was touched. WDBC (§5.2) is entirely unreproduced.
3. **Non-Gaussian designs.** All designs are Gaussian or Gaussian-plus-masking. The paper's
   heavy-tailed (t₃, App F.5.2), lognormal and squared-map settings (`utils.gen_square_map`,
   `gen_lognormal`) were not run, and the oracle ν_j / ρ_j closed forms I rely on for claims 1, 2
   and 5b **only exist because the design is jointly Gaussian**. Outside that, my oracle arm has no
   analogue.
4. **High dimension.** Max p tested is 50 (claim 3) and 20–40 elsewhere, with n ≥ p throughout
   except p=40/n=100. The paper's high-dimensional regime (p=400, n=300, App F.5.3), where it
   claims knockoffs *fail* and SKO still holds, is untested.
5. **The plug-in algorithm.** Claims 1 and 2 are verified under the theorems' **oracle** hypothesis
   ("Given ν_j and ρ_j"). My plug-in runs (claim 1, arm B) showed type-I inflation of ~0.14–0.22
   at α=0.10 that did not decay up to n=1600. I did not determine whether this is (a) a genuine
   gap between the theory and Algorithm 1, (b) an artefact of my imputer choice (plain ridge,
   λ=1e-2, no cross-fitting), or (c) an artefact of fitting ν̂/ρ̂ and evaluating on the same data.
   **Claim 2's FDR result was never re-run with estimated ν̂/ρ̂** — the FDR guarantee under
   estimation is therefore untested here, and given the claim-1 arm-B result it should not be
   assumed to carry over.
6. **Theorems 4.2 and 4.3.** The 𝒲₁ control (Thm 4.2) was not measured; Thm 4.3 was not run at all.
7. **Test choice.** Only the sign test and Wilcoxon were used, per Algorithm 1/2. Other
   nonparametric paired tests were not explored.
8. **Monte-Carlo resolution.** Verdicts rest on 250–600 seeds; MC standard errors are recorded in
   each JSON. Differences smaller than ~2 s.e. should not be read as real. In particular claim 2's
   q=0.05 row (FDR 0.000, power 0.000) is vacuous and carries no information about control.
9. **Competing methods.** Only HRT was implemented as a comparator, in my own simplified form
   (50/50 split, K resamples, oracle conditional mean). dCRT, LOCO, Sobol-CPI, CPI and standard
   knockoffs — all present in Figures 4/5 — were not implemented, so "SKO is best" is not tested;
   only "SKO-5 beat my HRT in three configs" is.
10. **The authors' code was read, not executed.** `official_code/` requires sklearn/pandas/joblib.
    My implementations follow `src/semi_KO.py` and `src/utils.py` but are independent
    reimplementations; a discrepancy between them and the authors' runtime behaviour would not
    have been caught.

---

## Reproducibility gate

`check_reproducibility.py` re-asserts every number quoted above against `results/*.json`, and with
`--rerun` re-executes all six verify scripts and requires them to reproduce their stored artifacts
exactly. It is a bespoke gate for this logbook, **not** a project test suite (none exists here).

```
./.venv/bin/python check_reproducibility.py            # 39 assertions, ~1 s
./.venv/bin/python check_reproducibility.py --rerun    # + full re-execution, ~6 min
```

Status at time of writing: **39/39 assertions pass**; all six scripts (claims 1, 1b, 2, 3, 5, 5b)
were re-executed from scratch and reproduced their JSON **bit-for-bit** apart from `wall_seconds`,
confirming the seeding is deterministic and no reported number is stale.

Note that the gate deliberately guards the *unfavourable* findings too — the claim-5b caveat
`power(SKO-1) < power(HRT)` is an assertion, so it cannot be quietly dropped by a later edit.

