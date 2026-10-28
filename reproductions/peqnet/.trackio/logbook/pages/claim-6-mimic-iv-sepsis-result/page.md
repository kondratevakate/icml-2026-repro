# Claim 6: MIMIC-IV sepsis result


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_92fdbf8f2354", "created_at": "2026-07-29T15:43:54+00:00", "title": "Claim 6: MIMIC-IV sepsis result"}
-->
**Paper claim.** "In a MIMIC-IV cohort of 999 adult ICU sepsis patients with hypotension, higher MAP targets for vasopressor weaning are associated with higher serum lactate at 72 hours."

**Verdict - INCONCLUSIVE.**

The paper reports 999 adult ICU patients satisfying Sepsis-3 criteria,
hypotension (`MAP <= 65 mmHg`), and vasopressor initiation within 24 hours. It
does not release cohort SQL, MIMIC item IDs, missing-data rules, preprocessing,
model code, checkpoints, or random seeds.

The local protected-data download is not complete. Reconstructing a different
cohort and comparing it to the paper figure would not verify the anchored
claim. This page will remain inconclusive until the exact cohort path is
auditable. No patient-level data are included in this logbook.

```json
{
  "cohort_sql_released": false,
  "complete_mimic_iv_available": false,
  "item_id_mapping_released": false,
  "preprocessing_and_seeds_released": false
}
```


---
<!-- trackio-cell
{"type": "code", "id": "cell_a1a6f5559a5c", "created_at": "2026-07-29T15:43:54+00:00", "title": "C6 machine-readable evidence", "language": "python"}
-->
````output
{
  "blockers": {
    "cohort_sql_released": false,
    "complete_mimic_iv_available": false,
    "item_id_mapping_released": false,
    "preprocessing_and_seeds_released": false
  },
  "verdict": "INCONCLUSIVE"
}
````
