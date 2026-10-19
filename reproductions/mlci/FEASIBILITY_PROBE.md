# Feasibility probe: A Machine-Learned Comorbidity Index

Probe date: 2026-07-29.

Paper: arXiv 2606.17450, OpenReview `C6ZTjSXbz7`.

## Artifact state

- The arXiv source archive is public and was inspected directly.
- Archive SHA-256:
  `ef1745380be34e0b3cf2d16f5412ec6c22f7d7cb395b8bff2d8f6143d460fe10`.
- The source contains the complete method specification, theorem proofs,
  result tables, and aggregate risk-curve CSVs.
- No official training repository, executable entrypoint, patient-level
  learned scores, or model checkpoint was found in the paper source or
  targeted search.
- The current leaderboard snapshot has zero attempts for this paper.
- `C:\Projects\data\digitaltwin` does not yet contain the full MIMIC-III or
  MIMIC-IV core tables. Only demos and unrelated event-log datasets are
  currently visible there. The readiness scanner was rerun after the cohort
  implementation and still reports all eight release/table gates as blocked.

## Verdict

**GO in two phases.**

1. Now: independently audit the Section 5 DeepSets+nHSIC construction and the
   Lemmas 6.1-6.2 / Theorem 6.3 chain on CPU.
2. After download: build the exact admission cohorts and rerun the MIMIC
   tables. Do not publish table claims from the paper's TeX values or aggregate
   risk-curve CSVs.

The independent C1 and C5 audits now pass. Synthetic end-to-end cohort
fixtures also pass for both releases, including ICD filtering, outcome labels,
ICU timing, code normalization, and patient split leakage checks.

## Data gate

For MIMIC-IV, the required raw modules are:

- `hosp/admissions`
- `hosp/patients`
- `hosp/diagnoses_icd`
- `icu/icustays`

For MIMIC-III, the required raw tables are:

- `ADMISSIONS`
- `PATIENTS`
- `DIAGNOSES_ICD`
- `ICUSTAYS`

The paper restricts MIMIC-IV to admissions with at least one ICD-10 diagnosis.
It uses ICD prefixes of length four, patient-level train/validation/test
splits, and four outcomes: in-hospital mortality, 30-day mortality, length of
stay above seven days, and ICU transfer. MIMIC-III uses a late-transfer ICU
label with first ICU entry more than 24 hours after admission.
