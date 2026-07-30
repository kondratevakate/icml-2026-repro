# Claim 4: synthetic task-shift evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ceb2b5a218f3", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 4: synthetic task-shift evidence"}
-->
**PARTIAL - 1/2.** Adaptive toy training and BALD expert inference
execute on CPU after creating the output directory and enabling UTF-8.
Three runs with the same declared seed span 0.0745 validation AUROC because
the entrypoint does not seed training RNGs. This is not the canonical
30-run experiment.


---
<!-- trackio-cell
{"type": "code", "id": "cell_c0282010dfb3", "created_at": "2026-07-30T08:08:15+00:00", "title": "Claim 4: synthetic task-shift evidence evidence", "language": "python"}
-->
````output
{
  "command_scope": "Official sequence/adaptive toy path, one epoch, one MC sample, one inner update, seed argument 999, CPU.",
  "attempt_1": {
    "validation_auroc": 0.7188,
    "outcome": "failed because the entrypoint did not create results/"
  },
  "attempt_2": {
    "validation_auroc": 0.7276,
    "average_test_auroc": 0.616,
    "outcome": "metrics saved, then cp1252 failed on final emoji print"
  },
  "attempt_3_utf8": {
    "validation_auroc": 0.6531,
    "average_test_auroc": 0.5666037752884746,
    "average_test_auprc": 0.38805493598625074,
    "outcome": "completed with PYTHONUTF8=1"
  },
  "same_seed_validation_auroc_range": 0.07450000000000001,
  "interpretation": "The random_seed argument controls data splitting but method/main.py does not seed torch or NumPy, so training is not repeatable.",
  "expert_smoke": {
    "mode": "BALD",
    "targets": 5,
    "queries_per_target": 2,
    "svi_steps": 20,
    "svi_steps_per_query": 20,
    "completed": true
  }
}
````
