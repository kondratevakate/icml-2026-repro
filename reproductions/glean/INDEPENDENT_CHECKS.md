# Independent checks

## Outcome

Prepared score: **2/12**, equal to the frozen forecast.

## C1 - Partial (1/2)

The packaged synthetic check independently evaluates the discounted logit
accumulation and its recurrence form. The equations are internally
implementable.

The Appendix A bound is not valid as written. Approximate sufficiency assumes

`E[(p_S(S) - p*(tau))^2] <= epsilon_suff`,

but the final bound substitutes `2 * epsilon_suff^2`. The derivation supports
`2 * epsilon_suff`, not its square. For `0 < epsilon_suff < 1`, the printed
term is strictly smaller and cannot follow from the assumption. This is a
substantive falsification of the stated bound, while leaving the empirical
method possible.

## C2 - Partial (1/2)

All official MIMIC-IV-Ext CDM v1.1 payload checksums match. The pathology map
contains the documented 2,400 cases:

- appendicitis: 957;
- cholecystitis: 648;
- diverticulitis: 257;
- pancreatitis: 538.

The exact three GLEAN diseases are therefore present. The v1.1
`radiology_reports.csv` has 5,960 nonempty, unique-note rows, one more than the
5,959 stated on the official dataset page. This is an upstream release/summary
discrepancy, not evidence about GLEAN's generated trajectories.

The paper-specific 4,000 trajectories, 50/50 balancing decisions, case
sampling, correctness labels, and judge outputs are not released.

## C3 - Unsupported (0/2)

Table 1 can be transcribed and arithmetically compared but not regenerated.
Against Self-Consistency averaged over all six backbone/disease cells, active
GLEAN improves mean AUROC by 12.25% and reduces mean Brier by 52.60%, which
explains the abstract's approximate headline.

If "best baseline" is interpreted independently per cell, mean relative AUROC
improvement is only 9.24%, while Brier reduction is 50.39%. The paper does not
state the aggregation convention.

More importantly, the Main Results prose conflicts with Table 1 for the
Qwen3-30B diverticulitis example:

| Quantity | Prose | Table 1 |
| --- | ---: | ---: |
| GLEAN K=3 AUROC | 0.9789 | 0.9794 |
| GLEAN K=3 Risk@0.5 | 0.0802 | 0.0741 |
| GLEAN K=3 Brier | 0.0647 | 0.0632 |
| Active AUROC | 0.9856 | 0.9862 |
| Active Risk@0.5 | 0.0494 | 0.0370 |

Without row-level outputs, neither version can be adjudicated.

## C4 - Unsupported (0/2)

The ablation tables are internally readable but no triggered cases, retrieved
guidelines, judge scores, or per-ablation predictions are released.

## C5 - Unsupported (0/2)

Best-of-N values and compute totals are display-only. There are no candidate
groups, rankings, call traces, or token logs.

## C6 - Unsupported (0/2)

Only aggregate clinician-study statistics and two short qualitative excerpts
are provided. No anonymized ratings, error-step labels, study instrument, or
analysis code is released.

