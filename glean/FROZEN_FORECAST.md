# Frozen forecast

Locked at `2026-07-30T11:11:28+04:00`, after inventorying the paper, source,
public guideline metadata, the credentialed MIMIC-IV-Ext CDM release, and the
absence of an author code or output release. The forecast was fixed before the
packaged audit and final scoring run.

## Forecast

**2/12 points**, plausible range **1-4/12**.

| Claim | Forecast | Pre-check basis |
| --- | ---: | --- |
| C1 method and error-bound analysis | 1/2 | Equations and proof are public, but no implementation is released. |
| C2 data and trajectory construction | 1/2 | Both source datasets are accessible, but the 4,000 generated trajectories are absent. |
| C3 main verification results | 0/2 | No predictions, labels, calibration split, seeds, or code. |
| C4 active/component ablations | 0/2 | Only rendered tables and figures are available. |
| C5 Best-of-N and compute trade-off | 0/2 | No candidate groups, verifier scores, or token/call logs. |
| C6 clinician study | 0/2 | No ratings, annotations, protocol, or analysis code. |

The main upside is an independently checkable theoretical derivation and
complete access to the two source datasets. The main downside is that neither
source dataset contains the paper's generated agent trajectories or verifier
outputs.

