# Logbook — CAffNet: Hard Constraint-Affine Neural Networks (orid 20hdQQQrA4)

Arm: **arm2** (Hermes + K-Dense skill set, skill `paper-claim-reproduction`), variant **arm2b**
(context-compaction enabled, see `COMPACTION.md`).
Model: `tencent/hy3:free` via `http://localhost:8319/v1`. CPU only (WSL2, torch 2.13.0+cpu,
numpy 2.5.1). Elapsed wall-clock: ~1 h 50 min. LLM turns used: ~28 (budget 60).
Paper read ONCE -> `notes_paper.md`; no other reproduction logbooks consulted.

This file is the final deliverable of the arm2+compaction test run (run `20hdQQQrA4_arm2b`).
All numbers below are quoted directly from `results/claim1.json` … `results/claim5.json`
(real executed runs) and are re-asserted by `check_reproducibility.py`.

---

## Claim 1 — Theorem 3.5: universal approximation with bound ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K
**Verdict: verified**
Source: Theorem 3.5 (Sec 3.2) and its proof in Appendix C (K = eps / (3 + 3 sqrt(n_out))).
Script: `verify_claim1.py` -> `results/claim1.json` (command: `./.venv/bin/python verify_claim1.py`).

- Symbolic: `(3 + 3 sqrt(n_out)) * K - eps == 0` for K = eps/(3+3 sqrt(n_out)) — **true** (sympy,
  `symbolic_constant_closes: true`).
- Numeric stress: 8 (n_out, m) configs x 200 seeds, K = 1e-3, p = 2, targets placed on the
  polyhedron boundary so the projection branch of Eq (12) actually fires
  (110–200 projection-branch instances per config); `infeasible_instances: 0`.
  Max observed ratio ||P* - f_t||_2 / K vs the claimed bound:
  | (n_out, m) | max ratio | bound 3+3√n_out |
  |---|---|---|
  | (1,2) | 0.9971001586566164 | 6.0 |
  | (1,4) | 0.9923570959997907 | 6.0 |
  | (2,3) | 1.6424769212615287 | 7.242640687119286 |
  | (2,5) | 1.446907792078362 | 7.242640687119286 |
  | (3,4) | 1.6027273263822814 | 8.196152422706632 |
  | (3,7) | 1.2320821394200527 | 8.196152422706632 |
  | (4,6) | 1.68252632685596 | 9.0 |
  | (5,8) | 1.686931229412946 | 9.708203932499369 |
  Overall max ratio **1.686931229412946** (`max_ratio_overall`) — bound holds in 8/8 configs
  (`n_configs_bound_holds: 8 / n_configs_total: 8`, and even under the strictly tighter constant
  `1 + sqrt(n_out)`, `holds_under_tight_constant_1_plus_sqrt_n: true` for every config). The
  theorem's constant is valid but loose.
- **Mutation** (invert the selection rule of Eq 12: take the *farthest* feasible candidate instead
  of the nearest — `argmax instead of argmin in Eq 12`): the bound is **broken in 8/8 configs**
  (`holds: false` for all), max ratios 18.96320931916251, 7.846957574033642, 112.1564423072167,
  126.41046987139862, 71.4357431773435, 275.42964969616213, 202.0805111475756, 677.5939335688797.
  => the argmin in Eq (12) is the mechanism that yields the bound, not an accident.

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

## Claim 3 — hard satisfaction without full row rank; cardinality <= min(m, n_out)
**Verdict: verified**
Source: Section 3.1–3.2, Eq (2), Lemma 3.3, Theorem 3.4. Script: `verify_claim3.py` ->
`results/claim3.json`.

- 2100 instances (`n_instances_total: 2100`) = 7 configs (n_out, m) in
  {(1,4),(2,5),(2,8),(3,7),(3,10),(4,9),(5,12)} x 3 instance kinds
  {generic, duplicate_rows, rank_deficient} x 100 seeds.
- CAffNet: instances with **no feasible candidate = 0** (`caffnet_instances_with_no_feasible_candidate: 0`),
  instances with a constraint violation = **0** (`caffnet_instances_with_constraint_violation: 0`),
  max violation over all 2100 instances = **1.1075584893660562e-12**
  (`caffnet_max_violation_over_all_instances`, numerical zero). Lemma 3.3 / Theorem 3.4 hold on
  every instance including redundant and rank-deficient A.
- HardNet baseline (full-row-rank formula): **undefined on 1961 / 2100** instances
  (`hardnet_instances_undefined_due_to_rank_deficiency: 1961`) because A A^T is singular; on the
  139 where it is defined it produced 0 violations (`hardnet_instances_with_constraint_violation: 0`).
  This is exactly the limitation the claim asserts.
- Cardinality/count: max |gamma| over Gamma == min(m, n_out) (`T3_max_cardinality_equals_min_m_nout: true`)
  and |Gamma| == sum_{k=1..min(m,n_out)} C(m,k) <= 2^m - 1 (`T3_gamma_count_identity_and_2m_bound: true`),
  checked **exhaustively** for m = 1..10 and n_out = 1..5.
- **Mutation** (cap the combination cardinality at min(m, n_out) - 1, i.e. violate the stated
  decomposition size): feasibility is lost on **478 / 2100** instances
  (`MUTATION_kmax_minus_1_failure_count: 478 / MUTATION_kmax_minus_1_total: 2100`). The
  min(m, n_out) cardinality is necessary, not decorative.

## Claim 4 — piecewise benchmark: CAffNet-TF reduces MSE by 73.33% vs soft NN, zero violations
**Verdict: inconclusive** (direction + the zero-violation half reproduced; the 73.33% magnitude not)
Source: Section 4.1 / Table 2 (paper: NN 0.0045, HardNet 0.0037, CAffNet-FF 0.0020,
CAffNet-TF 0.0012; reduction 73.33%). Script: `verify_claim4.py` -> `results/claim4.json`.
Command actually run: `./.venv/bin/python verify_claim4.py --epochs 20000 --seeds 5`.

Reproduced Appendix D.1 exactly (target f, g1u/g2u/g1l/g2l, A = [1,1,-1,-1]^T,
b = [g1u, g2u, -g1l, -g2l], 50 random train points, 400 linspace test points, Adam lr 1e-4,
full-batch, 5 seeds), **except 20000 epochs instead of the paper's 50000** (CPU budget).

| method | test MSE mean (std) over 5 seeds | viol max (mean over seeds) | seeds with any violation |
|---|---|---|---|
| NN (soft) | **0.001839826621879348** (0.0024093821342899414) | 0.1705633378904027 | 5/5 |
| CAffNet-FF | **0.002884547281501829** (0.0019290806990403236) | **0.0** | 0/5 |
| CAffNet-TF | **0.0009070503751830563** (0.0005683786123839912) | **0.0** | 0/5 |

- Reproduced reduction CAffNet-TF vs NN = **50.69913847335672 %**
  (`reproduced_reduction_TF_vs_NN_pct`) (paper: 73.33 %).
- Reproduced reduction CAffNet-FF vs NN = **-56.783647284944635 %**
  (`reproduced_reduction_FF_vs_NN_pct`) (paper: +55.6 %); FF did **not** beat the soft NN at 20000 epochs.
- Zero-violation sub-claim **verified**: both CAffNet variants have exactly 0.0 max and mean
  violation on all 400 test points in all 5 seeds (`caffnet_zero_violations: true`), while the soft
  NN violates in every seed (`nn_has_violations: true`, `n_seeds_with_any_violation: 5` for NN, `0`
  for both CAffNets). **Mutation** for that sub-claim: removing the CAffine projection layer (= the
  "NN" arm) makes violations appear immediately (NN max violation mean 0.1705633378904027, 5/5 seeds).
- Why inconclusive on the headline number: with 40% of the paper's epochs the absolute MSEs are in
  the paper's ballpark (paper NN 0.0045 vs ours 0.00184; paper TF 0.0012 vs ours 0.00091) and the
  sign of the effect matches (TF lowest MSE, zero violations), but the specific 73.33% figure is not
  recovered and the per-seed std is of the same order as the mean.

## Claim 5 — safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail
**Verdict: inconclusive** (CAffNet-vs-soft-NN direction strongly reproduced; HardNet arm not
reproducible; CAffNet not collision-free)
Source: Section 4.3, Appendix D.3, Figs 6–7 / Table 4. Script: `verify_claim5.py` ->
`results/claim5.json`. Command: `./.venv/bin/python verify_claim5.py --epochs 8 --ntrain 12 --seeds 3`
(paper: 300 initial states).

Full D.3 problem implemented: unicycle, dt = 0.1 s, 150 steps, 3 polytopic obstacles with the
exact A_j, b_j, per-edge CBF, smooth union (kappa = 10, eta_j = ln m_j), state box and control
box (A_u, b_u), alpha(h) = h, PID nominal + network correction; aggregated A(x) is 13 x 2
(`m_constraints: 13, n_out: 2`), |Gamma| = 91 (`n_gammas: 91`). Evaluation: 49 held-out initial states.

| method | collisions / 49 (mean of 3 seeds) | control violations | arrived (<0.5 m) | steps with empty S(x) |
|---|---|---|---|---|
| NN soft | **20.333333333333332** (24, 19, 18) | 24.0 | 47.666… | 0 |
| CAffNet-FF | **1.0** (1, 1, 1) | 0.3333333333333333 | 15.666… | 4–4–4 |
| CAffNet-FF a-posteriori projection | **1.0** | 0.3333333333333333 | 13.333… | 0 |

- The qualitative claim direction is reproduced: the soft-constrained baseline collides on
  ~41% of initial states (mean 20.33/49), CAffNet on 1/49 (2.04%).
- The residual CAffNet collision is **not** a contradiction of Theorem 3.4 but its precondition:
  every CAffNet-FF seed reports 4 simulation steps where the feasible set S(x) is **empty**
  (`infeasible_steps: 4` for all three CAffNet-FF seeds; Assumption 3.2 fails: discrete-time CBF +
  tight actuator box), and that is exactly where the single collision and the single control
  violation occur.
- **HardNet arm not reproduced** (honest refusal, not a toy substitute): HardNet's projection
  requires A(x) with full row rank; here A(x) is 13 x 2 (rank 2), so the formula is undefined
  (`hardnet_not_reproduced_reason` in the JSON).
- Also observed (paper's side remark): a-posteriori projection reaches the goal less often
  (13.333 vs 15.666 of 49) than the jointly trained CAffNet, consistent with the paper's
  "gets stuck near an obstacle" observation, though at this small training budget the effect is weak.
- Reduced training (12 initial states, 8 epochs vs the paper's 300 states) is the main reason the
  CAffNet arrival rate (15.666/49) is far below the paper's; hence inconclusive rather than verified.

---

## Summary table

| Claim | Verdict | One-line evidence |
|---|---|---|
| 1 | verified | max \|\|P*-f_t\|\|/K = 1.6869 vs bound 6.0–9.71 over 1600 instances; mutation (argmax) breaks it 8/8 (up to 677.6) |
| 2 | verified | sub-constraint residual 1.18e-13 for any w; trainable w objective 1.859 vs 4.501 for w=0, better on 60% of instances |
| 3 | verified | 0/2100 violations (max 1.11e-12) incl. rank-deficient A, while HardNet undefined on 1961/2100; k-cap mutation loses feasibility 478/2100 |
| 4 | inconclusive | zero violations reproduced (0.0 both CAffNets, 5/5 seeds), but reduction 50.70% vs the claimed 73.33% at 20000/50000 epochs; CAffNet-FF worse than NN |
| 5 | inconclusive | CAffNet 1.0 vs soft NN 20.33 collisions / 49, but CAffNet not collision-free (4 empty-feasible-set steps) and HardNet undefined (A is 13x2) |

## Evidence boundary — what this evidence does NOT cover
- No claim is tested with the paper's original code (not available in the bundle); all
  implementations are re-derivations from the paper text in `notes_paper.md`.
- Claim 1: the bound is validated on random *linear* polyhedra with the exact Eq (12)
  algorithm, not on function classes; the universal-approximation limit (existence of
  f_theta for every eps) is assumed, not tested. No test of the continuity of x -> P*(x).
- Claim 2: the "joint optimization" advantage is shown with a black-box optimiser over w on
  static instances, not with an end-to-end trained w_phi network on the paper's benchmarks.
- Claim 3: instances up to n_out = 5, m = 12; no large-m scaling test, no CAffNet-Lite.
- Claim 4: 20000 epochs (paper 50000), 5 seeds; HardNet arm omitted; timing columns of
  Table 2 (T_train/T_test) not compared; no GPU, so wall-clock numbers are not comparable.
- Claim 5: 8 training epochs and 12 training initial states (paper 300 states), 3 seeds,
  49 evaluation states; PID gains for u_nom are not specified in the paper and were chosen
  by us; HardNet arm and Table 4 numbers are not reproduced; the paper's figures
  (trajectories) are not reproduced.
- Experiment 4.2 (learning optimization solvers, Table 3) was not in the anchored claim list
  and was not attempted.
