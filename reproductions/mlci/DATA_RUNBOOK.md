# MLCI MIMIC data runbook

This runbook starts only after the protected MIMIC tables have finished
downloading. Raw patient-level data must remain local and must never be
committed or uploaded to the Hugging Face Space.

## 1. Readiness gate

From `mlci/repro_mlci`:

```powershell
python inspect_mimic_tables.py `
  --root C:\Projects\data\digitaltwin `
  --output results\mimic_readiness.json
```

Continue only when the relevant release is marked `ready`. The scanner accepts
CSV, compressed CSV, and Parquet files and checks the required columns.

MIMIC-IV needs:

- `hosp/admissions`
- `hosp/patients`
- `hosp/diagnoses_icd`
- `icu/icustays`

MIMIC-III needs:

- `ADMISSIONS`
- `PATIENTS`
- `DIAGNOSES_ICD`
- `ICUSTAYS`

## 2. Build MIMIC-IV cohort

Replace the four paths with the exact files found by the readiness report:

```powershell
python build_mimic_cohort.py `
  --release mimic-iv `
  --admissions C:\path\to\mimic-iv\hosp\admissions.csv.gz `
  --patients C:\path\to\mimic-iv\hosp\patients.csv.gz `
  --diagnoses C:\path\to\mimic-iv\hosp\diagnoses_icd.csv.gz `
  --icustays C:\path\to\mimic-iv\icu\icustays.csv.gz `
  --output-dir C:\Projects\data\digitaltwin\derived\mlci\mimic-iv
```

Paper targets:

- 254,377 admissions and 122,905 patients
- admission split: 178,178 / 25,228 / 50,971
- patient split: 85,980 / 12,235 / 24,690

## 3. Build MIMIC-III cohort

```powershell
python build_mimic_cohort.py `
  --release mimic-iii `
  --admissions C:\path\to\MIMIC-III\ADMISSIONS.csv.gz `
  --patients C:\path\to\MIMIC-III\PATIENTS.csv.gz `
  --diagnoses C:\path\to\MIMIC-III\DIAGNOSES_ICD.csv.gz `
  --icustays C:\path\to\MIMIC-III\ICUSTAYS.csv.gz `
  --output-dir C:\Projects\data\digitaltwin\derived\mlci\mimic-iii
```

Paper targets:

- 58,976 admissions and 46,520 patients
- admission split: 41,200 / 5,867 / 11,909
- patient split: 32,488 / 4,630 / 9,402

## 4. Mandatory checks

Each build writes a cohort, diagnosis tokens, and a JSON report. Before model
training:

1. `status` must be `passed`.
2. `unique_admissions` and `no_patient_split_leakage` must be true.
3. Overall admission and patient counts must be reconciled before interpreting
   model differences.
4. Split-count mismatches alone are not a cohort failure because the paper
   does not disclose its deterministic hash algorithm or salt.
5. Run sensitivity splits with at least three documented hash choices if the
   exact published split cannot be recovered.

## 5. Remaining protocol gaps

The paper does not disclose:

- deterministic patient-split hash algorithm and salt;
- size and sampling rule for the fixed bandwidth-estimation subsample;
- an unambiguous implementation of the task-weight clip-factor operation.

The bundle uses SHA-256 with an empty salt by default and records the choice.
No C2-C4 or C6 verdict should be published until these choices are documented,
the comparator panel is trained on the same splits, and fresh test metrics are
saved.
