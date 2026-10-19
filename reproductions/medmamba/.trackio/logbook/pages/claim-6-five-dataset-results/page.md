# Claim 6: five-dataset results


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1dd1b0eac98b", "created_at": "2026-07-29T16:40:28+00:00", "title": "Claim 6: five-dataset results"}
-->
**Verdict - INCONCLUSIVE, NOT EXECUTED.**

The five datasets and seeds 41-45 were not run. The author repository releases
only an APAVA shell script, delegates data preparation to another repository,
and includes no checkpoints or cached metrics. Paper tables are not treated as
reproduction evidence.


---
<!-- trackio-cell
{"type": "code", "id": "cell_897e4845fba7", "created_at": "2026-07-29T16:40:28+00:00", "title": "C6 machine-readable evidence", "language": "python"}
-->
````output
{
  "id": "C6",
  "paper_anchor_present": true,
  "reason": "No fresh five-dataset, five-seed execution was performed.",
  "released_dataset_scripts": [
    "scripts/APAVA_Subject.sh"
  ],
  "required_datasets": [
    "ADFTD",
    "APAVA",
    "PTB",
    "PTB-XL",
    "TDBRAIN"
  ],
  "score": 0,
  "verdict": "INCONCLUSIVE_NOT_EXECUTED"
}
````
