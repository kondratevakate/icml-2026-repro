# Reproduction report — SALaD (0xJt4PqPJv)

**Paper:** *Explicitly Modeling Censoring Produces Superior Survival Predictors* — Shi-ang Qi, Yakun Yu, Russell Greiner (ICML 2026).
**Code:** github.com/shi-ang/SALaD (cloned verbatim into `repo_ref/`).
**Reproducer:** `reproduce.py` (CPU, `.venv`, torch 2.13, SurvivalEVAL 0.8.0).

## Verdict

**REFUTED.** Explicitly modeling censoring does not produce a superior survival
predictor in the regime the paper targets. In the coupled semi-synthetic regime
(semi-SUPPORT, where the event and censoring times share latent factor h2), the SALaD
latent-decomposition variant only matches the plain baseline (ΔC-index ≈ 0, reproduced
on two independent seed bases); on real SUPPORT it loses to the plain baseline on every
one of 5 paired splits. Where a censoring-aware structure does help at all, it is the
two-branch variant — not SALaD.

## What was tested

For one survival layer (N-MTLR) three representation structures were compared under an
identical pipeline, over 5 paired random splits (same seed → same train/test split):

- `N-MTLR`        — plain baseline (models event time only)
- `N-MTLR-2B`     — two separate branches (event + censoring)
- `N-MTLR-salad`  — shared `kappa` + event-specific `epsilon` + censor-specific `gamma`
  representations, with cosine-orthogonality regularization (`alpha=10`) and an
  IPM/MMD balancing term on the shared representation (`beta=0.01`), `e-dims=c-dims=16`
  (the README's documented configuration).

Datasets: **SUPPORT** (9105 patients, open, ~32% censored) and **semi-SUPPORT**
(same covariates, semi-synthetic times where event ~ factors {h1,h2} and censoring
~ {h2,h3}; the shared factor h2 makes censoring dependent — the exact "coupled" regime
the paper predicts SALaD should exploit; evaluated against the true event time).

## Results (mean ± sd over 5 splits; paired deltas)

| Dataset | Model | C-index ↑ | IBS ↓ | MAE-Hinge ↓ |
|---|---|---|---|---|
| SUPPORT | N-MTLR (baseline) | 0.6885 ± 0.0050 | 0.1589 ± 0.0090 | 354.7 |
| SUPPORT | N-MTLR-2B | **0.6921** ± 0.0051 | **0.1567** ± 0.0096 | **343.5** |
| SUPPORT | N-MTLR-salad | 0.6833 ± 0.0044 | 0.1616 ± 0.0090 | 388.2 |
| semi-SUPPORT | N-MTLR (baseline) | **0.7456** ± 0.0077 | 0.0049 ± 0.0017 | **71.00** |
| semi-SUPPORT | N-MTLR-2B | 0.7442 ± 0.0048 | 0.0049 ± 0.0017 | 70.98 |
| semi-SUPPORT | N-MTLR-salad | 0.7437 ± 0.0040 | 0.0049 ± 0.0017 | 71.04 |

Paired (same-seed) SALaD − baseline:

- **SUPPORT:** ΔC = −0.0052 (SALaD better in **0/5** seeds), ΔIBS = +0.0027 (1/5),
  ΔMAE-Hinge = +33.4 (0/5). Consistent loss.
- **semi-SUPPORT:** ΔC = −0.0019 (1/5), ΔIBS = 0.0000 (tie), ΔMAE-Hinge = +0.04 (0/5).
  Statistical tie.

## Evidence boundary

- **Independently replicated.** An independent audit re-ran the pipeline on a disjoint
  seed base (seeds 100–102): SUPPORT SALaD − baseline ΔC = negative on 3/3 splits;
  semi-SUPPORT ΔC ≈ −0.002 (tie). The null result is seed-robust, not a lucky seed.
- **One layer, two datasets.** The paper claims gains across 4 survival layers and 10
  datasets. This reproduction covers the N-MTLR layer on SUPPORT + semi-SUPPORT, which is
  the most directly testable coupled case; it does not measure other layers (e.g. AFTNN)
  or datasets, so it does not exclude gains there.
- **Hyperparameters** follow the repository README's documented SALaD configuration
  (`alpha=10, beta=0.01, e-dims=c-dims=16, neurons=64,64, dropout=0.6`). Per-dataset
  tuning was not performed; the paper may report tuned settings.
- **One deviation from upstream:** predicted survival curves are clipped to [0,1] before
  evaluation, a numerical guard against float32 rounding (observed max 1.0000004) that
  the strict SurvivalEVAL 0.8.0 validator rejects. The repo does not pin SurvivalEVAL;
  this removes a version-mismatch artifact and does not affect the science.
- Training converges (loss plateaus well before the early-stop patience of 50), so the
  null result is not undertraining.

## Reproduce

```bash
cd salad_censoring
.venv/Scripts/python.exe reproduce.py --data SUPPORT      --family N-MTLR --n-exp 5 --out runs/support_nmtlr.json
.venv/Scripts/python.exe reproduce.py --data semi-SUPPORT --family N-MTLR --n-exp 5 --out runs/semisupport_nmtlr.json
```
