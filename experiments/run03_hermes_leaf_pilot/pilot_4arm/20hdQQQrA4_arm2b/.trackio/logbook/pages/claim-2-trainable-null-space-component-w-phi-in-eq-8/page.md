## Claim 2 — trainable null-space component w_phi in Eq (8)
**Verdict: verified**
Source: Eq (8), Section 3.2 discussion, Remark 3.6. Script: `verify_claim2.py` -> `results/claim2.json`.

- 3 configs (n_out, m) in {(2,3),(3,5),(4,6)} x 40 seeds = 120 instances (`n_instances: 120`),
  360 w-perturbation checks (`T1_n_checks: 360`).
- Structural: max |A_gamma P_gamma - b_gamma| over all instances and all random w =
  **1.1834977442504169e-13** (`T1_max_subconstraint_residual_over_all_w`) — the sub-constraint is
  met exactly for *any* w (the extra term really lies in the null space).
- Non-degeneracy: 86 rank-deficient A_gamma cases (`T2_rank_deficient_cases: 86`); min / mean
  output spread when varying w = **1.530619010628436 / 2.732280847454395**
  (`T2_min_output_spread_over_w` / `T2_mean_output_spread_over_w`, so w moves the output along a
  face); in **34/34** full-column-rank cases `T2_full_rank_cases_with_zero_nullspace: 34` the term
  I - A_gamma^+ A_gamma = 0 exactly, matching the paper's statement that the null-space term vanishes.
- Optimisation value (the "joint optimization across multiple feasible projections" part): objective
  ||y - y*||^2 minimised over w vs the fixed orthogonal projection (w = 0):
  mean **1.859256323957484** (`T3_mean_obj_trainable_w`) vs **4.500871503391247**
  (`T3_mean_obj_w_zero_orthogonal_projection`); median gap **0.11952442114037204**
  (`T3_median_gap`), max gap **26.87171091351931** (`T3_max_gap`); trainable w strictly better on
  **60%** (`T3_frac_instances_trainable_w_strictly_better: 0.6`) of instances.
- **Mutation** (delete the null-space term, w forced to 0): objective is never better —
  `MUTATION_w_forced_zero_is_never_better: true`; it is strictly worse in 60% of instances.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 trainable null-space component w_phi in Eq (8)"}\n-->
