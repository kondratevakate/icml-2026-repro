# Claim decomposition: diabetes safety generalization

## Frozen scorecard

| Claim | Reproduction route | Verdict | Score |
| --- | --- | --- | ---: |
| C1: unified simulator supports three diabetes-treatment environments and the released benchmark covers eight algorithms | Run official tests and all environment IDs; inventory released policies | Partially verified: simulator fully executes, but the eight-algorithm benchmark was not rerun | 1/2 |
| C2: safe policies exhibit an ID-to-OOD safety-generalization gap | Independently reconstruct released checkpoints and compare patient 1 with patients 2-10 over seven days | Partially verified for CPO, T1D adolescent, seed 0 | 1/2 |
| C3: BA-NODE outperforms ITransformer and NODE over a 24-step horizon | Train all three models on the public transition dataset across disclosed seeds and recompute MAE/FDE/RMSE | Not executed | 0/2 |
| C4: one-sided reliability plus an epsilon shield margin implies finite-horizon hypo/hyper safety | Audit the implication and destructive boundary mutations | Verified as a conditional theorem | 2/2 |
| C5: the released predictive shield implements the mechanism used for broad clinical gains | Audit train/load paths, patient normalization, and action pruning; execute only if contracts close | Partially verified at source level; released path has three material divergences | 1/2 |
| **Prepared** |  |  | **5/10** |

## Evidence boundaries

- C1 does not reproduce performance of all eight algorithms.
- C2 is a scoped mechanism reproduction under a corrected post-step metric
  trace, not an exact replay of the released evaluator fallback or the
  three-seed aggregate table.
- C3 remains zero. Paper-rendered metrics are not execution evidence.
- C4 verifies only the stated conditional implication. It does not establish
  that BA-NODE satisfies the reliability assumption.
- C5 does not claim that clinical gains are false. It establishes that the
  released shield cannot be connected unchanged to the released training
  output and does not hard-prune penalized actions.

## Stop condition

Do not spend local CPU time retraining the full predictor panel. Reopen C3 and
the broad C5 empirical claim only on persistent Linux/GCP after fixing the
documented path and cohort-index contracts in an explicitly labeled repair.
