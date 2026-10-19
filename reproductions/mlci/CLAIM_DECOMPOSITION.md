# Claim decomposition: A Machine-Learned Comorbidity Index

## Anchored claims

| Claim | Full-verdict route | Current state |
|---|---|---|
| C1 DeepSets encoder with weighted multi-outcome nHSIC | Independent implementation must pass permutation, masking, kernel, weighting, and gradient checks | Exact Appendix-width CPU audit passed: max permutation error `5.22e-8`, padding error zero, parameter-gradient norm `12.66` |
| C2 MIMIC-IV distance correlation 54.80 vs 36.41 | Exact ICD-10 cohort, patient split, model/baseline training, and test dCorr | Blocked on full MIMIC-IV and absent official code |
| C3 MIMIC-III distance correlation 39.06 vs 32.28 | Exact MIMIC-III cohort, split, training, and test dCorr | Blocked on full MIMIC-III and absent official code |
| C4 MIMIC-IV mutual information 74.22 vs 69.92 | Exact cohort, models, estimator, and seeds | Blocked on full MIMIC-IV and absent official code |
| C5 rank-one shared structure implies a monotone threshold rule | Audit binary-label rank-one identity, weighted alignment, SVD reduction, threshold certificate, and uniform transfer bound | CPU audit passed across 100 seeds; max identity errors about `2e-15`, zero split or bound failures |
| C6 rank-one energy and coincident thresholds on both cohorts | Recompute from patient-level labels and learned scores | Blocked on full cohorts and absent learned scores |

## Expected score

Do not lock a publication forecast yet.

- C5 has a plausible full two-point path after adversarial review.
- C1 is likely one or two points depending on whether the independent
  implementation is accepted as full rather than toy evidence.
- C2-C4 and C6 remain zero until generated from the user's MIMIC data.
- A theory-only publication would therefore have a `2-4/12` ceiling and is not
  worth publishing before the data phase.

## Stop conditions

- Stop empirical work if cohort counts cannot match the paper closely.
- Do not use the arXiv TeX tables as reproduction evidence.
- Do not claim C6 from the aggregate risk-curve CSVs; the required
  patient-level label matrix and learned score ordering are absent.
- Do not infer baseline performance without training the full comparator
  panel under the same patient split.
- Treat split-count differences separately from cohort-count differences:
  the paper gives 70/10/20 patient splits but does not disclose the
  deterministic hash algorithm or salt.
