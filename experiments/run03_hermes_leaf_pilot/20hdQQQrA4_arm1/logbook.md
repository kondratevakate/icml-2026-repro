# Reproduction logbook — CAffNet (Hard Constraint-Affine Neural Networks)

- **OpenReview id:** 20hdQQQrA4 · **arXiv:** 2605.24437
- **Run:** arm1 (Hermes leaf, CONTROL — no skills, isolated)
- **Agent:** Hermes leaf subagent (`delegate_task role=leaf`)
- **Model:** tencent/hy3:free via localhost:8319/v1 (local router)
- **Env:** numpy 2.5.1, scipy (via brainiac_brainage_py311 venv); torch 2.13.0+cpu for claim4
- **Generated:** 2026-08-02

## Per-claim verdicts (from executed artifacts, not text)

| Claim | Verdict | Key evidence (executed numbers) |
|-------|---------|----------------------------------|
| 1 | **verified** | n_bound_violations=0/6000; worst_ratio_err_over_bound=0.166 (<1); mutation_w_scale_50K also 0 violations |
| 1b | **verified** | subbound_intersection_gamma_violations=0; mutation_argmax_bound_violations=0 |
| 2 | **verified** | A_consistency max_abs_residual=1.87e-11 (~0); trained strictly better than orthogonal in 8/8 seeds (mean loss 0.458 vs 0.896) |
| 3 | **verified** | caffnet violating_instances=0, max_violation=9.2e-15 (~0); hardnet_like violates 962/2160; truncated variants violate |
| 4 | **falsified** | measured_reduction_pct=-10.08 vs paper claimed +73.33; CAffNet_TF MSE (0.564) WORSE than NN-soft (0.512) — improvement NOT reproduced |
| 5 | **inconclusive** | mechanism test passed (caffnet 0 collisions; hardnet_like/soft collide); but paper's trained Table-4 policies NOT reproduced (spec missing) |

## Notes
- Claims 1–3 verified with real executed numbers; claim 4 contradicts the paper (negative
  reduction) → falsified; claim 5 mechanism-level only, full policy not reproduced.
- This is the CONTROL arm: no K-Dense skills, no external framework. Directly comparable to
  arm2 (skills), arm3 (Sakana), arm4 (ARC-light) on the same paper/model.
- `torch` was required only for claim4 (CAffNet training); all other claims ran on numpy/scipy.
