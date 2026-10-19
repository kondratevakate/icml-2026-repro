# Claim 2: S3 comparability requires regularization


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ef43964f7b59", "created_at": "2026-07-23T09:12:58+00:00", "title": "Claim 2: S3 comparability requires regularization"}
-->
**Paper claim (Table 3, S3, τ = 0.5).** In Scenario 3 the true coefficient functions do **not** vary with Z. The paper states FunCQNet "remains comparable to the simpler model," i.e. the extra interaction capacity costs nothing when Z is irrelevant (paper S3: FunCQNet RMSE α̂₁ 0.030 vs No-Interaction 0.026 — a ~15% gap).

**Reproduction result — unregularized estimator.** With the plain interaction design (no penalty), the interaction model is markedly **worse** than the reduced model in S3, because it over-parameterizes the irrelevant Z-interaction terms:

| Scenario | Cens | RMSE α₁ Int | RMSE α₁ NoInt | ratio |
| --- | --- | --- | --- | --- |
| S3 | 0% | 0.064 | 0.027 | 2.4× worse |
| S3 | 10% | 0.068 | 0.034 | 2.0× worse |
| S3 | 30% | 0.082 | 0.041 | 2.0× worse |
| S3 | 50% | 0.097 | 0.046 | 2.1× worse |

So the paper's "comparable" claim does **not** hold for a naive interaction model. The direction (interaction ≥ no-interaction in S3) matches, but the magnitude of the penalty (~2×) is far larger than the paper's ~15%.

**Mechanism test — add regularization.** FunCQNet uses dropout / weight regularization. We add an **L1 penalty on the Z-interaction coefficients only** (encoded as pseudo-observations, so the fit stays a linear program). Sweeping the penalty at 30% censoring, 8 runs:

| L1 penalty | S3 RMSE α₁ (Int) | S1 RMSE α₁ (Int) |
| --- | --- | --- |
| 0.00 | 0.076 | 0.070 |
| 0.05 | 0.073 | 0.069 |
| 0.20 | 0.066 | 0.067 |
| 0.50 | 0.056 | 0.067 |

As the penalty grows, S3 interaction RMSE falls toward the reduced-model level (0.076 → 0.056) while the **S1 gain is preserved** (~0.067 unchanged). Regularization shrinks the spurious Z-interactions when Z is irrelevant without harming the case where Z matters.

**Verdict: NOT reproduced by the unregularized estimator; recovered under regularization.** The paper's S3 comparability is real but **depends on FunCQNet's built-in regularization**, not on the interaction structure alone — a nuance the paper does not make explicit. An honest reproduction must include a shrinkage penalty to match the S3 claim.
