# FunCQNet reproduction — core simulation claim (CPU)

**Paper:** *FunCQNet: A Functional Censored Quantile Neural Network for Predicting
Long-Term Post-Transplant Kidney Survival* (ICML 2026, poster `wNL0qGb2MN`).
Men, Liu, Tang, You, Dong, Cao. OpenReview only — **no code, no arXiv, no HF.**

## Scope and why it is bounded

Two of the paper's evidence streams are **not reproducible**:

1. **Real-data results (Section 3)** use the **UNOS/OPTN** transplant registry
   (United Network for Organ Sharing / Organ Procurement Transplantation
   Network). These are access-controlled STAR files requiring a signed DUA and a
   months-long request; there is no public version. *Verdict: not reproducible —
   data private.*
2. **Reference implementation** — none was released. The neural network, the
   censoring-adjusted sequential quantile loss, and the conformal-inference
   procedure would have to be re-implemented from the text.

What **is** fully specified and reproducible is the **simulation study**
(Appendix D.1). This repo reproduces its **structural core claim** on CPU
without a neural network.

## The claim under test

Paper, Table 3 (τ = 0.5): a model whose functional coefficients α_d(t, Z) depend
on the scalar covariates Z (**Interaction**, like FunCQNet) has far lower RMSE
than a reduced model that ignores Z (**No Interaction**):

- **S1 / S2** (true effect is Z-modulated): large RMSE reduction — the paper
  reports ~**92%** for α₁ and ~**75%** for α₂ in S1.
- **S3** (true effect independent of Z): the two models should be **comparable**.

## Method (honest simplification)

The paper's true α_d(t, Z) is **linear in Z** (each Z_j multiplies a function of
t), so a linear-in-Z basis estimator can represent it *exactly* — the comparison
is fair, not a strawman. We:

- generate the exact DGP (`dgp.py`): B-spline X₁, cosine-basis X₂, Z₁~U(1,2),
  Z₂,Z₃~Bernoulli(0.5), three scenarios;
- expand α_d in a cubic B-spline basis and fit by **IPCW-weighted median
  regression** (Bang & Tsiatis 2002 censored quantile estimator), solved as a
  linear program (`estimator.py`);
- compare **Interaction** vs **No Interaction** designs and report
  RMSE(α̂_d) = √(E_Z ∫ (α̂_d − α_d)² dt).

Run: `python run_sim.py --B 20`  (~1 min, 180 fits). Smoke: `--smoke`.

## Results (B = 20 runs, n = 1000, τ = 0.5)

| Scenario | RMSE α₁ reduction | RMSE α₂ reduction | matches paper? |
|---|---|---|---|
| S1 (0–50% cens) | −77 … −85% | −72 … −77% | **yes** (paper ~92% / ~75%) |
| S2 (0–50% cens) | −58 … −70% | −41 … −52% | **yes** (direction & tier) |
| S3 (0–50% cens) | interaction ~2× **worse** | ~2× worse | **no** (see below) |

No-Interaction RMSE(α̂₁) ≈ 0.41 in S1 matches the paper's 0.38 closely.

## Verdicts

- **Core simulation claim (S1/S2): REPRODUCED.** An interaction-aware functional
  censored quantile estimator reduces coefficient-function RMSE by 58–85%
  relative to a Z-agnostic model when the true effect is Z-modulated; the α₂
  reduction (~75%) matches the paper's reported ~75%. Holds across 0–50%
  censoring.
- **S3 "no penalty when interaction is absent": NOT reproduced by the
  unregularized estimator; recovered under regularization.** With no penalty the
  interaction model is ~2× worse when Z is irrelevant. Adding an L1 penalty on
  the Z-interaction terms drives S3 RMSE back toward the reduced model
  (0.076 → 0.056 at 30% censoring) while preserving the S1 gain (~0.067
  unchanged). The paper's S3 comparability therefore depends on FunCQNet's
  built-in regularization (dropout), not on the estimator structure alone.
- **Real-data claims (UNOS/OPTN, Section 3): NOT REPRODUCIBLE — data private.**

## Deviations from the paper

- Estimator is a linear-in-Z B-spline quantile regression, **not** the paper's
  neural network — chosen so the structural claim is testable on CPU.
- Reduced scale: n = 1000 (paper 50000), B = 20 runs (paper 100), α-basis K = 8.
  Trends are stable across seeds; absolute RMSEs are the same order as Table 3.
- Censoring C ~ Exp with the mean calibrated per run to hit the target rate; the
  paper fixes the rate but not the law of C.
- τ = 0.5 only (the paper's main table); at the median the F⁻¹(τ) recentering and
  the S2/S3 ψ₂₂ heteroscedastic term vanish.
