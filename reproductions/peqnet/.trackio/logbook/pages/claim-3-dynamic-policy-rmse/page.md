# Claim 3: dynamic-policy RMSE


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5845fad55a8d", "created_at": "2026-07-29T15:43:54+00:00", "title": "Claim 3: dynamic-policy RMSE"}
-->
**Paper claim.** "For dynamic policies that differ in only the first two steps or in all steps, PEQ-Net achieves lower RMSE than DeepLTMLE in both semi-synthetic DGPs (Table 2)."

**Verdict - INCONCLUSIVE.**

The table was not treated as self-validating evidence. No author PEQ-Net code
or author DeepLTMLE implementation was found in the paper artifacts. The run
also requires processed MIMIC-III trajectories, 500 training epochs, selected
hyperparameters, and 20 independent seeds.

The printed lag sum starts at `i=1` but contains `(-1)^i/(1-i)`, so its first coefficient divides by zero.

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
{"type": "code", "id": "cell_492650131b3c", "created_at": "2026-07-29T15:43:54+00:00", "title": "C3 machine-readable evidence", "language": "python"}
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
