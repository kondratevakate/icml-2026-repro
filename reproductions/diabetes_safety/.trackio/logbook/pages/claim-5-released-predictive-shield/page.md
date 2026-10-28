# Claim 5: released predictive shield


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_06c8ea435206", "created_at": "2026-07-29T20:21:46+00:00", "title": "Claim 5: released predictive shield"}
-->
**Verdict - PARTIALLY VERIFIED AT SOURCE LEVEL (1/2).**

The released code contains the predictive workflow, but its runnable contracts
do not close unchanged:

- training writes `ba_node_b...`, while the shield loads `fe_b5...`;
- adult, child, and adolescent patient 001 all resolve to normalization index
  `0` instead of `0/10/20`;
- a logit penalty of `10` leaves unsafe-action
  probability `4.540e-05` in a two-action
  counterexample, rather than hard pruning.

These findings limit reproducibility of the released shield. They do not
falsify the paper's clinical-gain tables.


---
<!-- trackio-cell
{"type": "code", "id": "cell_f13d729ac8e1", "created_at": "2026-07-29T20:21:46+00:00", "title": "C5 machine-readable evidence", "language": "json"}
-->
````output
{
  "finite_penalty": {
    "hard_pruning": false,
    "penalty": 10.0,
    "unsafe_action_probability": 4.5397868702434395e-05,
    "verdict": "FINITE_PENALTY_DOES_NOT_ENFORCE_THEOREM_PERMISSION_SET"
  },
  "release_source": {
    "cohort_indices_match": false,
    "cohort_literal_comparison_anchor": true,
    "expected_cohort_indices": {
      "adolescent#001": 20,
      "adult#001": 0,
      "child#001": 10
    },
    "finite_logit_penalty_anchor": true,
    "path_contract_matches": false,
    "released_cohort_indices": {
      "adolescent#001": 0,
      "adult#001": 0,
      "child#001": 0
    },
    "shield_folder_expression": "f'saved_files/dynamics_predictor/fe_b5_p24_h24/1'",
    "shield_source": "../official/GlucoAlg/shield/predictive_shield.py",
    "train_folder_expression": "f'saved_files/dynamics_predictor/ba_node_b{args.n_basis}_p{args.prediction_length}_h{args.history_length}/{args.seed}'",
    "train_source": "../official/GlucoAlg/2.train_dynamics_predictor.py",
    "verdict": "RELEASED_PREDICTIVE_SHIELD_NOT_RUNNABLE_AS_DOCUMENTED"
  }
}
````
