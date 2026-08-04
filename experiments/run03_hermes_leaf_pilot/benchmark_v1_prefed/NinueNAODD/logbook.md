# Reproduction logbook — NinueNAODD

**Paper:** *Efficient Bayesian Inference from Noisy Pairwise Comparisons* (BBQ)
Aczel, Theis, Wattenhofer. OpenReview `NinueNAODD` · arXiv:2510.09333 (**v2, 12 May 2026** — the version used for all source references below).

**Agent:** arm1, autonomous. **Hardware:** CPU only (8 cores, WSL2). **Stack:** Python 3.12, numpy 2.5.1, scipy 1.18.0, pandas/pyarrow. No GPU, no LLM calls.
**Global seed:** `20260802` (pinned in every script; all randomness via `np.random.default_rng(seed)`).
**Total wall clock:** ≈ 55 min of compute (well inside the 8 h budget).

## What was built

| file | role |
|---|---|
| `bbq_core.py` | From-first-principles BBQ EM (Eqs. 11–13), Bayes-BT (EM with `q≡1`), Crowd-BT (Chen et al. 2013-style SGD), Eq. (3) log-likelihood, Gamma/Beta log-prior, Kendall's tau, synthetic generator from Eq. (2) |
| `ihq_data.py` | Loader for the real IHQ dataset (`Mabyduck/CLIC2024-test-human-eval`, screened/unscreened parquet, downloaded from HF) |
| `verify_claim1..6.py` | One verifier per anchored claim; each writes `results/claim<N>.json` |
| `results/claim<N>.json` | Numeric results, protocol, mutation test, verdict |
| `data/{screened,unscreened}.parquet` | Real IHQ data (only external data used) |

Hyperparameters throughout are the paper's (Appendix B): Gamma prior `a=5, b=0.1`, Beta prior `α=10, β=2`, `ELO = 400·log λ`, convergence when no Elo score moves by more than 1.

## Data reality check (affects claims 3, 4, 6)

The public HF release is **smaller** than what Appendix D Table 6 reports:

| split | comparisons (HF) | raters (HF) | comparisons (paper) | raters (paper) |
|---|---|---|---|---|
| IHQ-screened | 1,920 | 48 | 2,012 | 50 |
| IHQ-unscreened | 1,921 | 59 | 2,062 | 62 |
| IHQ-all | 3,841 | 107 | 4,074 | 112 |

Also, the **gt** used for IHQ in the paper is the *official CLIC 2024 leaderboard*, which is not redistributed. We fall back to the paper's own alternative gt ("the ranking achieved on the whole dataset"). All three models do agree on the top item (the uncompressed `reference`), consistent with the paper's statement. HUMAINE, MT-Bench, WD, HiFiC and ConHa are not obtainable here; Google's Crowd-BT implementation was likewise not used — ours is a reimplementation.

## Verdicts

| # | Claim (short) | Source | Verdict |
|---|---|---|---|
| 1 | Closed-form EM ⇒ monotone likelihood + stationary point (vs. Crowd-BT) | Sec. 3.3, Eqs. 11–13, App. A | **verified** |
| 2 | BBQ converges in seconds; Crowd-BT ≈ 15 min on HUMAINE | Fig. 3 / Sec. 4.4 | **inconclusive** (BBQ half verified) |
| 3 | IHQ-unscr. top-1: 61.92 / 33.15 / 24.32 | Table 1 | **inconclusive** (not reproduced; ordering also not reproduced) |
| 4 | 1st in τ on 5/8, 2nd on the other 3; 100% top-1 on MT-Bench/WD/HiFiC | Table 1, Sec. 4.1 | **falsified** (as a conjunction, against the paper's own table) |
| 5 | 99% Type I error ≈1% (BBQ, Crowd-BT), ≈0.1% (Bayes-BT) | App. F/G, Fig. 4(v2) | **inconclusive** |
| 6 | q_r vs rater agreement, Pearson r = 0.724 (IHQ-unscr.) | Sec. 4.3, Fig. 2, Eq. 2 | **inconclusive** (direction & significance verified, magnitude off) |

---

## Claim 1 — — verified

**Verdict:** sec. 3.3, eqs. 11–13, app. a

*Source:* Sec. 3.3, Eqs. (11)–(13); derivation App. A. *Script:* `verify_claim1.py` (15 s).

- **100 EM runs** (25 synthetic datasets × 4 random inits, data drawn from Eq. 2): **0 monotonicity violations** of the log-posterior; worst single-step change `+3.6e-3` (i.e. always increasing).
- **Stationarity** at convergence: max |∂ log-posterior| = `5.7e-8` (relative `5.9e-11`); EM fixed-point residual `2.4e-12` (λ), `5.6e-14` (q).
- **Contrast:** our Crowd-BT gradient run decreases its own objective on **16 of 39** epoch transitions (worst step −2.40) — no monotonicity, as the paper argues.
- **Mutation:** replacing the E-step responsibility γ (Eq. 11) with a mis-specified `clip(γ^0.35·1.4)` breaks monotonicity in **10/10** datasets (worst step −17.26). ✔ property is load-bearing.
- *Caveat:* this is an empirical check of an analytic guarantee, not a proof. The guarantee is the standard EM result (Dempster et al. 1977) and the M-step updates are exact maximisers of Q, so this is expected.

## Claim 2 — — inconclusive (BBQ half verified)

**Verdict:** fig. 3 / sec. 4.4

*Source:* Fig. 3 / Sec. 4.4; dataset shapes App. D Table 6. *Script:* `verify_claim2.py` (25 s).

- BBQ wall-clock, plain NumPy: IHQ-scr **0.069 s** (121 it), IHQ-unscr **0.098 s** (115 it), IHQ-all **0.393 s** (200 it), and a synthetic **HUMAINE-shaped** set (104,781 comparisons, 1,977 raters, 27 items) **1.38 s** (25 it). → "converges within seconds" ✔ on everything tested.
- The "≈15 minutes for Crowd-BT" figure is implementation-specific. Our Crowd-BT costs **4.6 s/epoch** on the HUMAINE-shaped set (≈4.6 min for 60 epochs) — same order of magnitude, but it is *not* Google's code and not the real HUMAINE data, so this neither confirms nor refutes the number.
- **Mutation:** subsampling to 20 raters (~1% of data) cuts BBQ runtime 47× (1.38 s → 0.029 s), confirming the timing is data-size driven rather than a fixed cost. ✔

## Claim 3 — — inconclusive (not reproduced)

**Verdict:** table 1

*Source:* Table 1, IHQ-unscr. top-1. *Script:* `verify_claim3.py` (19 min, 500 rater-bootstraps).

| | BBQ | Crowd-BT | Bayes-BT |
|---|---|---|---|
| task spec | 61.92 | 33.15 | 24.32 |
| arXiv v2 Table 1 | 61.42 | 32.44 | 23.59 |
| **ours** | **44.20** | **71.40** | **23.20** |

- Bayes-BT lands almost exactly on the paper's value (23.2 vs 23.59). BBQ is well below, and **our Crowd-BT is the most stable top-1 identifier — the opposite of the claim's ordering.**
- Confounders that stop this being a refutation: smaller public data sample; substitute gt; our Crowd-BT is a reimplementation whose stability is hyperparameter-dependent (lr 0.02, 20 epochs); 500 instead of 10,000 bootstraps.
- **Mutation:** re-drawing every pair's winner by a fair coin collapses all three to 4.0%, i.e. chance level (1/28 = 3.57%). ✔ the metric is measuring real signal.
- Version note: the task spec's numbers (61.92/33.15/24.32) are **not** the v2 numbers (61.42/32.44/23.59) — they come from an earlier version of the paper.

## Claim 4 — — falsified (as stated)

**Verdict:** table 1, sec. 4.1

*Source:* Table 1 + Sec. 4.1. *Script:* `verify_claim4.py` (15 min).

Ranking the three methods by Kendall's τ using the paper's **own** Table 1 numbers:

`HUMAINE 1 · MT-Bench 2 · WD 2 · HiFiC 1 · ConHa 1 · IHQ-all 1 · IHQ-scr **3** · IHQ-unscr 1`

- "first on 5 of 8" → **true** (5 firsts).
- "100% top-1 on MT-Bench, WD, HiFiC" → **true** per Table 1 (not independently reproduced — datasets unavailable).
- "second on the remaining three" → **false**: BBQ is second on only 2, and **third on IHQ-screened** (BBQ 0.9204 < Bayes-BT 0.9211 < Crowd-BT 0.9238). The conjunctive claim is false against its own source table.
- **Mutation** (audit): shaving 0.01 off BBQ's HUMAINE τ drops the first-place count 5 → 4. ✔ the count is sensitive, not a tautology.
- Partial empirical check on the 3 IHQ splits (200 bootstraps, τ vs the full-data BBQ ranking — a reference that *favours* BBQ): BBQ best on IHQ-all (0.9320), but Crowd-BT best on IHQ-scr (0.9135 vs 0.9098) and IHQ-unscr (0.8758 vs 0.8704). Consistent with the paper's own finding that BBQ is not first on IHQ-screened.

## Claim 5 — — inconclusive

**Verdict:** app. f/g, fig. 4(v2)

*Source:* App. F (interval construction) + Fig. 4 (v2); task cites App. G Fig. 6. *Script:* `verify_claim5.py` (9 min; 1,000 trials/config instead of 10,000, 50 comparisons per rater, R ∈ {5, 20, 50}).

**Finding: the paper's stated interval constant is inconsistent with its stated confidence level.** Appendix F gives `p99 = sqrt(diag(cov)) · 3.29`, but `sqrt(2)·erfcinv(0.001) = 3.2905` is the **99.9%** two-sided constant; the 99% one is 2.576. Mean Type I error over R ∈ {5,20,50}:

| rule | BBQ | Bayes-BT | Crowd-BT |
|---|---|---|---|
| paper's literal non-overlap w/ 3.29 | 0.00% | 0.00% | 0.07% |
| non-overlap w/ 2.576 | 0.03% | 0.00% | 1.13% |
| 99% interval on the Elo **difference** | 1.60% | 1.13% | 1.13% |

- Under the standard difference test, BBQ and Crowd-BT are indeed ≈1% ✔ — but **Bayes-BT is also ≈1%, not the claimed ≈0.1%**. Under the paper's literal recipe *everything* is ≈0.0–0.1%. No single decision rule reproduces the claimed 1% / 1% / 0.1% pattern simultaneously → inconclusive.
- The qualitative "Bayes-BT is the most conservative" ordering does hold under every rule.
- **Mutation:** with a true win probability of 0.60 (H₀ false), rejection rates jump to 97–100% for all models and rules. ✔ the ~1% is genuinely a null-hypothesis quantity, and the tests have power.
- *Caveat:* the credible-interval construction for BBQ/Bayes-BT (conditional Gamma posterior at the EM fixed point) is ours; the paper does not specify it fully, so the exact Bayes-BT number is implementation-sensitive.

## Claim 6 — — inconclusive (direction verified, magnitude off)

**Verdict:** sec. 4.3, fig. 2, eq. 2

*Source:* Sec. 4.3, Fig. 2, Eqs. (2) and (12). *Script:* `verify_claim6.py` (5 s). Real IHQ data.

| split | our r | n (ours) | paper r | n (paper) |
|---|---|---|---|---|
| unscreened | **0.855** (p = 6.8e-18) | 59 | 0.724 | 62 |
| screened | **0.755** (p = 5.5e-10) | 48 | 0.551 | 50 |

- Strong, highly significant positive correlation between the fitted rater-quality parameter q_r and each rater's agreement with the final BBQ ranking, on both splits, with the same pattern as Fig. 2 (**unscreened > screened**). Fitted line on unscreened: `q = 0.519·agreement + 0.500` (paper: `0.348·x + 0.631`).
- Magnitudes are ~0.13–0.20 higher than the paper's. Plausible causes: the smaller public sample, and the paper not specifying its "agreement with the final ranking" statistic (we use the fraction of a rater's comparisons whose winner outranks the loser under the BBQ ranking on the same split).
- **Mutation:** permuting q_r across raters (1,000 permutations) gives mean |r| = 0.106, max 0.493, permutation p = 0.000. ✔ the correlation is not an artefact of the marginals.
- The *practical* part of the claim — that q_r flags unreliable raters without a separate screening step — is supported: on the unscreened split q_r spans 0.60–0.95 and tracks agreement (0.31–0.97) monotonically.

---

## Summary of honest limitations

1. **5 of 8 datasets** (HUMAINE, MT-Bench, WD, HiFiC, ConHa) are not obtainable → claims 2, 3, 4 can only be checked partially or against the paper's own table.
2. **The official Crowd-BT implementation** (Google) was not used; ours is a faithful-in-spirit but not identical SGD reimplementation. Any Crowd-BT number here is ours, not the paper's.
3. **The public IHQ release is smaller** than the paper's own dataset table, so exact percentages/correlations on IHQ are unreachable by construction.
4. **The IHQ gt** (CLIC 2024 leaderboard) is unavailable; the paper's fallback gt was used instead.
5. Bootstrap/trial counts were reduced (500 / 200 / 1,000 instead of 10,000) to fit the CPU budget; all reported quantities are stable to the third digit under this reduction except claim-3 top-1 percentages, which carry ≈±2 pp Monte-Carlo error.
6. Two **paper-internal inconsistencies** surfaced and are reported rather than smoothed over: the 3.29 vs 2.576 confidence constant (claim 5) and the "second on the remaining three" statement contradicted by Table 1 (claim 4).

## Reproducing

```bash
cd <this dir>
python3 -m venv .venv && .venv/bin/pip install numpy scipy pandas pyarrow
.venv/bin/python verify_claim1.py    # ~15 s
.venv/bin/python verify_claim2.py    # ~25 s
.venv/bin/python verify_claim3.py    # ~19 min
.venv/bin/python verify_claim4.py    # ~15 min
.venv/bin/python verify_claim5.py    # ~9 min
.venv/bin/python verify_claim6.py    # ~5 s
```
Data is already in `data/`; re-download with
`curl -sL -o data/unscreened.parquet https://huggingface.co/datasets/Mabyduck/CLIC2024-test-human-eval/resolve/main/data/unscreened-00000-of-00001.parquet` (and likewise `screened`).