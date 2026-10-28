# scRatio independent release audit

This bundle audits the official artifacts for **Flow-Based Density Ratio
Estimation for Intractable Distributions with Applications in Genomics**
(OpenReview `5zbPdMNcl9`, arXiv `2602.24201v2`). It never uses leaderboard
scores or third-party verdicts.

The audit separates four evidence levels:

1. a fresh analytic-oracle check of the released ratio ODE;
2. a seeded 1,200-step CPU Gaussian smoke run;
3. independent aggregation of tracked CSV and executed-notebook outputs; and
4. explicit blockers where raw predictions, checkpoints, or paper-scale
   retraining are absent.

Local audit command (inside the prepared environment):

```bash
python audit_artifacts.py --runtime --small-training
pytest -q test_audit_artifacts.py
```

The paper-scale experiments are not claimed as rerun. The official Zenodo
record contains processed datasets, while the clone does not track checkpoints
or the external `project_folder/results` arrays referenced by application
notebooks.
