# MLCI publication preflight

Status: **HOLD - empirical data gate is not met**.

## Passed locally

- C1 Appendix-width DeepSets and weighted nHSIC construction audit.
- C5 rank-one and monotone-threshold proof-chain audit across 100 seeds.
- MIMIC-IV synthetic cohort integration path.
- MIMIC-III synthetic cohort integration path.
- ICD filtering and normalization checks.
- Mortality, 30-day mortality, long-stay, and ICU-transfer label checks.
- Patient-disjoint split and leakage check.

## Blocking publication

- Full protected MIMIC-IV core tables are not present locally.
- Full protected MIMIC-III core tables are not present locally.
- C2-C4 and C6 have no fresh empirical evidence.
- No official code or checkpoint was found.
- The paper does not disclose the split hash/salt or complete bandwidth
  subsampling details.

## Release decision

Do not create the canonical Hugging Face Space yet. A theory-only submission
has an estimated ceiling of 2-4/12 and would consume the one canonical attempt
without testing the paper's main medical claims.

Release only after:

1. At least one real cohort count is reconciled.
2. The matching C2/C3/C4 metrics are generated from fresh runs.
3. Raw MIMIC rows and identifiers are excluded from the bundle.
4. The final bundle contains aggregate metrics, scripts, hashes, environment,
   seeds, and an explicit limitations statement.
