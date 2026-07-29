# Claim 2: expanded-confounding RMSE


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0130a524c48f", "created_at": "2026-07-29T15:43:54+00:00", "title": "Claim 2: expanded-confounding RMSE"}
-->
**Paper claim.** "Across all three deterministic-policy contrasts in the expanded time-varying-confounding semi-synthetic DGP, PEQ-Net achieves lower RMSE than DeepLTMLE (Table 1)."

**Verdict - INCONCLUSIVE.**

The table was not treated as self-validating evidence. No author PEQ-Net code
or author DeepLTMLE implementation was found in the paper artifacts. The run
also requires processed MIMIC-III trajectories, 500 training epochs, selected
hyperparameters, and 20 independent seeds.

The expanded recurrence is specified, but it delegates treatment and outcome generation to the same undefined limited-DGP equations.

The paper's numerical values are therefore preserved only as provenance. A
faithful independent comparison cannot be executed from the released
artifacts, so no performance conclusion is claimed.

Machine-readable readiness:

```json
{
  "author_code_found": false,
  "printed_dgp_executable": false,
  "processed_mimic_iii_available": false,
  "selected_hyperparameters_and_seeds_reported": false
}
```


---
<!-- trackio-cell
{"type": "code", "id": "cell_7881157528b0", "created_at": "2026-07-29T15:43:54+00:00", "title": "C2 machine-readable evidence", "language": "python"}
-->
````output
{
  "all_requirements_available": false,
  "blockers": {
    "author_code_found": false,
    "printed_dgp_executable": false,
    "processed_mimic_iii_available": false,
    "selected_hyperparameters_and_seeds_reported": false
  },
  "verdict": "INCONCLUSIVE"
}
````
