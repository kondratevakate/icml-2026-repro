# SurvFD repair bundle

CPU-only repair targeting two judge findings:

- Claim 1 was previously tested only on the log-hazard scale. The new audit
  performs an exact functional-ANOVA decomposition on log-hazard, hazard, and
  survival functions.
- Claim 6 was not attempted. The new audit compares regression and permutation
  interaction estimators on four synthetic tasks with 10-16 features. This is
  explicitly a toy-scale proxy for the paper's four-dataset scalability panel.

Run:

```powershell
python run_all.py
```

The previous independent evidence for Claims 2-4 is retained in
`prior_reproduction/`.
