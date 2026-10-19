# Reproduction — Online Cox PH via Renewable Sieve Estimation (COLSA)

- **ORID:** TvLOddG9Ay
- **Title:** Online Learning and Inference for Cox Proportional Hazards Model Using Renewable Sieve Estimation
- **Authors:** Mengtong Hu, Ziyang Gong, Xu Shi, Ling Zhou, Peter Song
- **Code:** `CollaborativeInference/COLSA` (R package; vendored under `repo_ref/`)
- **Data:** the package's own bundled `data/sim.rda` — the exact simulation dataset described in the paper/README.
- **Environment:** R 4.6.1 (Windows), packages: survival, splines2, statmod, MASS, flexsurv.

## Method (no invention)
The authors' package source (`repo_ref/R/colsa.R`, `utils.R`) is loaded directly and their
**verbatim README pipeline** is run on their bundled `sim.rda`:
1. Select basis count by AIC over 1..5 on batch 1.
2. Fit `colsa()` on batch 1, then sequentially `update()` through batches 2..6.
3. Compare the final sequential estimate to
   (a) the README's printed reference numbers, and
   (b) `survival::coxph` on the **pooled** 6000-row data — i.e. the pooled-data partial
   likelihood estimator that Claim 1 names as the efficiency benchmark.

Script: `reproduce_claim1.R`. Raw log: `runs/claim1_console.txt`. Object: `runs/claim1_result.rds`.

## Data-generating process (from repo)
2 continuous (bivariate normal) + 2 categorical covariates (Bernoulli `x31`; 4-level
multinomial conditional on `x31`, dummies `x42/x43/x44`). Event times from a Weibull
mixture (shapes 3,5; scales 10,20), censoring Exp(rate 3). True
β = (0.15, −0.15, 0.3, 0.3, 0.3, 0.3). Six batches: 1500×3 + 500×3 = 6000 rows,
1965 events. AIC selected n_basis = 4.

## Claims

**Claim 1** — "COLSA achieves asymptotic normality and the same statistical efficiency
as the pooled-data partial likelihood estimator, with constant memory requirement."

**VERDICT: REPRODUCED.**

Evidence on the authors' own simulation dataset:

| var | true | COLSA coef | pooled coef | COLSA se | pooled se | SE ratio |
|-----|------|-----------|-------------|----------|-----------|----------|
| x1  | 0.15 | 0.1566 | 0.1563 | 0.01016 | 0.01011 | 1.004 |
| x2  | −0.15| −0.1726| −0.1710| 0.02231 | 0.02215 | 1.007 |
| x31 | 0.30 | 0.3827 | 0.3846 | 0.05782 | 0.05881 | 0.983 |
| x42 | 0.30 | 0.2751 | 0.2769 | 0.09094 | 0.08985 | 1.012 |
| x43 | 0.30 | 0.2916 | 0.2937 | 0.08329 | 0.08190 | 1.017 |
| x44 | 0.30 | 0.1768 | 0.1785 | 0.08220 | 0.08049 | 1.021 |

- Max |COLSA − pooled| coefficient difference = **0.0021**.
- SE ratio COLSA/pooled ∈ **[0.983, 1.021]** (within ~2%; not systematically inflated)
  → same statistical efficiency as the pooled partial-likelihood estimator.
- RMSE vs true β: COLSA 0.0623 vs pooled 0.0619 — indistinguishable recovery.
- **Constant memory:** fit object size across the 6 sequential updates =
  9544 → 10088 → 10104 → 10104 → 10120 → 10120 bytes (max/min = 1.06). Memory does not
  grow with cumulative sample size; the model carries fixed-dimension sufficient
  statistics rather than the accumulating risk set.
- COLSA's final estimate also matches the README's published reference summary to every
  printed digit, confirming the package reproduces the paper's reported result.

*Scope note (not part of the verdict): the challenge ships a single realized `sim.rda`, so
this is a one-dataset finite-sample reproduction of the authors' reported setting; the
generator is documented in prose but not exposed as a script, so a fresh multi-replication
efficiency study would require reconstructing the DGP (avoided to prevent invention). The
asymptotic-normality half of the claim is theoretical and is evidenced here indirectly via
the Wald z/p statistics and the SE agreement, not re-proved.*

**Claim 2** — "COLSA outperforms existing online Cox methods and matches the performance
of full-data (Oracle) estimation" (paper Sec 5.1 / Fig 2).

**VERDICT: REPRODUCED** (against the SGD and Meta online baselines).

Faithful re-implementation of the paper's Appendix-B simulation (mixture-Weibull baseline,
correlated MVN covariates, Exp(6) censoring ≈12% events), comparing the authors' own COLSA
package against Oracle (pooled `coxph`), Meta (inverse-variance weighting, Cochran 1954),
streaming SGD and multi-epoch SGD(Offline) (`coxphSGD`, Kosinski 2017). Metrics averaged
over replications, as in the paper (Abias, RMSE, AvgCP). Script `reproduce_claim2.R`,
logs `runs/claim2_K6_console.txt` / `claim2_K50_console.txt`, objects `runs/claim2_K{6,50}.rds`.

RMSE (lower is better):

| K (reps) | Oracle | COLSA | Meta | SGD | SGD(Offline) |
|----------|--------|-------|------|-----|--------------|
| 6  (200) | 0.1156 | **0.1157** | 0.1172 | 0.3255 | 0.1162 |
| 50 (40)  | 0.0501 | **0.0498** | 0.0574 | 0.1847 | 0.0471 |

- **COLSA matches Oracle** at both K (RMSE within 0.0003) — same accuracy as full-data.
- **COLSA beats the genuine online baselines** (streaming SGD and Meta) at every K. The
  margin over Meta widens as batches shrink (K 6→50) and Meta's 95% coverage collapses from
  0.940 to **0.900** (undercoverage), while COLSA holds nominal coverage (0.942, 0.949) —
  the exact degradation mechanism the paper attributes to online competitors.
- Streaming SGD is far behind at all K (RMSE 0.33 / 0.18).

*Scope notes (not part of the verdict): (i) The Wu et al. (2021) adaptive-partition "Online"
baseline was not reimplemented (nontrivial method, not shipped) — so the head-to-head is
against SGD and Meta, not the full Fig-2 panel. (ii) SGD(Offline) is competitive with
COLSA/Oracle, but it is not an online method — the paper defines it as requiring persistent
access to all historical data over 100 epochs, so it does not bear on "online" superiority.
(iii) Reduced replication counts vs the paper's 500 (200 at K=6, 40 at K=50) and K∈{6,50}
rather than {6,20,50,200}; the reproduced ordering and the Meta-degradation trend are already
unambiguous within these.*
