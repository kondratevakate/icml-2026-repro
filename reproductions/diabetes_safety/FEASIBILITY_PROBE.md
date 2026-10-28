# Feasibility probe: diabetes safety generalization

Paper: *Safety Generalization Under Distribution Shift in Safe Reinforcement
Learning: A Diabetes Testbed*.

OpenReview: `kSUGLBHd0T`. arXiv: `2601.21094`.

Probe date: 2026-07-29.

## Primary artifacts

- GlucoSim commit:
  `f5662ccca607f11fd3bf30fde513ae0cf2ce420e`.
- GlucoAlg commit:
  `50d3134fd6f0e01019295be0d6b99f110f794753`.
- Paper PDF SHA-256:
  `2ec58592a9a1556c20142753d1487a7d9c0ac606e1e2f73fe8f9b96f3e4b8158`.
- arXiv source SHA-256:
  `66496281fe0fa6000bb7a3dda0759ecb4bffb5ee84ae831a022d5fb27da5722f`.
- The public `safe-diabetes-benchmark/transition_datasets` dataset contains
  180 NPZ files covering 90 virtual patients.

No leaderboard, peer reproduction, or third-party verdict was inspected or
used as evidence.

## Execution probe

- The official GlucoSim suite passes all 22 tests on CPU.
- Independent 288-step runs execute `t1d-v0`, `t2d-v0`, and
  `t2d_no_pump-v0`.
- Repeated runs with seed `20260729` produce identical trajectory hashes for
  all three environments.
- All environments expose 14-dimensional observations and
  `MultiDiscrete([5, 5])` actions.
- A released CPO checkpoint was reconstructed independently from its config
  and state dictionary, without importing vendored OmniSafe.
- One 7-day T1D adolescent rollout on the training patient has 100% TIR and
  risk index 1.12. The mean over patients 2-10 is 89.00% TIR and risk index
  3.10, a -11.00 percentage-point TIR gap and +1.99 risk gap.
- These metrics use one post-step glucose value per action. This is a corrected
  trace contract: the released evaluator requests `info["cgm"]`, but GlucoSim
  does not provide that key and the code therefore falls back to the pre-step
  observation. The result is not labeled as an exact replay of that fallback.

## Released-shield blockers

The released predictive-shield path is not runnable unchanged:

1. `2.train_dynamics_predictor.py` writes BA-NODE to
   `ba_node_b<n>_p<p>_h<h>/<seed>`, while `predictive_shield.py` loads the
   unrelated hard-coded path `fe_b5_p24_h24/1`.
2. Runtime passes names such as `child#001`, but cohort offsets compare the
   full string to `child` and `adolescent`. Adult, child, and adolescent
   patient 001 therefore all select normalization index 0 instead of
   0, 10, and 20.
3. The theorem assumes actions outside the permission set are pruned. The
   implementation uses a finite logit penalty, leaving unsafe actions with
   nonzero sampling probability.

These findings limit claims about the released implementation. They do not
prove that the paper's reported clinical improvements are false.

## Decision

`GO` for a conservative CPU/source audit. Do not train BA-NODE or claim the
full 72-setting shield table locally.

Prepared forecast: **5/10**.
