# Frozen forecast

Frozen after pinning and reading the paper/source/repository, but before
installing dependencies or executing released code.

- Expected score: **5 / 12**
- Plausible range: **3-7 / 12**

## Rationale

The official repository contains all five cohorts' five-fold label splits,
prototype clustering/GMM extraction, the downstream evidential model and loss,
metric code, and an example notebook. This should permit deterministic checks
of split integrity, component-prototype mechanics, belief/plausibility
identities, mixture weighting, and a small synthetic optimization path.

The release contains no WSI tiles, UNI2-h features, GMM embeddings,
prototypes, checkpoints, predictions, summary metrics, or run logs. Therefore
the headline C-index/IBS/IBLL tables, pathologist assessment, ablations, and
runtime comparisons cannot be independently recovered locally.

Static inspection also finds a potential non-default-lambda mismatch: the
paper specifies `lambda * Bel + (1-lambda) * Pl`, while the training loss
appears to multiply both survival terms by `lambda`. This is prespecified for
an independent numeric audit rather than treated as a verdict.

## Expected claim scores

| Claim | Forecast |
|---|---:|
| Five-cohort data and evaluation protocol | 1 |
| Dual-prototype evidential fusion | 2 |
| GRFN survival bounds and mixture | 1 |
| Headline discrimination | 0 |
| Calibration and uncertainty | 1 |
| Interpretability, ablations, and robustness | 0 |
