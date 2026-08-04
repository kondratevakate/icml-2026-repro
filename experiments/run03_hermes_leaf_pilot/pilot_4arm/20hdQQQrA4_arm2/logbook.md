# Reproduction logbook — CAffNet: Hard Constraint-Affine Neural Networks

- **OpenReview id:** 20hdQQQrA4 · **arXiv:** 2605.24437
- **Arm:** arm2 (Hermes agent + K-Dense skill set; method skill `paper-claim-reproduction`)
- **Run:** `20hdQQQrA4_arm2`, re-run after the previous arm2 attempt (`deleg_85700d01`)
  exhausted its tool budget on claim 4 and produced no logbook, gate or run metadata.
- **Environment:** Python 3.12.3, numpy 2.5.1, scipy 1.18.0, CPU only (8 vCPU, WSL2, no GPU).
  No torch. No official code executed (none is referenced as released in the claim bundle).
- **Self-contamination note:** the method skill ships a prior reproduction logbook for this
  exact paper (`references/20hdQQQrA4-caffnet-verification.md`). It was **deliberately not
  opened**, per §0 of the skill. No other arm's artifacts in this repository were read.

All five verifiers are independent numpy reimplementations of the paper's Section 3
equations (`caff_core.py`, `vec_caff.py`). Every number below is re-asserted by
`check_reproducibility.py`.

## Verdict summary

| Claim | Source | Verdict | Headline evidence |
|---|---|---|---|
| 1 | Theorem 3.5 (UAT + error bound `(3+3√n_out)K`) | **verified** | 0 / 2400 instances exceeded the bound; max ratio **1.71** vs the smallest applicable bound 8.196; 0 infeasible outputs; **706** instances exercised the projection branch |
| 2 | Section 3, trainable null-space term `w_phi` | **verified** | invariance 7.9e-14, reachability 2.2e-14, `w=0` reproduces the fixed orthogonal projection with diff **0.0**, `w` beats it on 100% of **600** instances (median loss reduction 0.517) |
| 3 | Section 3, no full-row-rank + cardinality `min(m,n_out)` | **verified** | 0 violations over **705** cases, all **705** rank-deficient; HardNet-Aff violates **57** of them (8.09%, max 17.67); truncating the enumeration to `min(m,n_out)−1` fails on **114** of 450 (25.33%) |
| 4 | Experiments, piecewise benchmark: 73.33% MSE reduction, zero violations | **verified (structural) / toy (magnitude)** | CAffNet 0 violations on 1200 test points, soft baseline **413**; measured reduction only **9.22%** at CPU scale vs the paper's 73.33% (which is internally consistent: 1 − 0.0012/0.0045 = 73.333%) |
| 5 | Experiments, safety-critical control | **partial: verified / inconclusive** | CAffNet: 0 collisions and ≤7.1e-15 constraint violation in all 3 regimes; baselines lose the hard guarantee (HardNet-Aff 0.620, soft 1.364) but **no baseline collided**, so the collision contrast is untested |

## Claim 1 — Theorem 3.5

Route: structural guarantee (skill §2l) — the theorem quantifies over the layer, not over a
trained network, so no training is needed. A feasible target `f_t` is perturbed by
`||η||_∞ ≤ K` to emulate an unconstrained approximator, pushed through the CAffine layer,
and `||P(f_θ) − f_t||₂ / K` is compared against `3 + 3√n_out`.

Fixed modest setting (budget rule): 200 trials × 3 seeds × 4 `(m, n_out)` cells × 3 values
of K = **2400** instances. No sweep.

- `n_bound_violations = 0`, `max_ratio_over_all = 1.7137897456544158`
- `n_infeasible_caffnet = 0`
- Branch coverage (P25): `n_case2_projected = 706` — half the instances place `f_t` exactly
  on an active constraint so the projection branch is genuinely exercised.
- **Mutation** (HardNet-Aff single pseudo-inverse on the identical instances): **92**
  infeasible outputs, max violation 5.762. Its error ratio never exceeded the bound, which
  is recorded rather than glossed: the mutation separates the arms on *feasibility*, not on
  the approximation constant.

Verdict: **verified**.

## Claim 2 — the trainable null-space component `w_phi`

Route: four-property decomposition (skill P26), in the equality geometry where the formula
is stated exactly; 600 under-determined instances (3 seeds × 200).

| Property | Measured |
|---|---|
| (a) invariance — `A P = b` for arbitrary `w` | max residual **7.85e-14** |
| (b) reachability — solve for the `w` hitting an arbitrary feasible target | max error **2.18e-14** |
| (c) degenerate — `w = 0` equals the fixed orthogonal projection | max diff **0.0** (exact) |
| (d) benefit — optimising over the feasible set beats the fixed foot | **100%** of instances, median relative loss reduction **0.517** |

**Mutation** (delete the term, `w := 0`): reproduces the fixed orthogonal projection
*exactly* on **600 / 600** instances — an exact equality, which is what proves nothing else
contributes. Verdict: **verified**.

## Claim 3 — rank deficiency and constraint cardinality

Route: exhaustive enumeration over a deterministic dictionary of constraint geometries
containing duplicated / negated / rescaled rows, plus a random rank-deficient sweep.

- `n_cases = 705`, `n_rank_deficient = 705`, `n_violations_caffnet = 0` (max violation 0.0).
  The degeneracy count is reported alongside the violation count: zero violations would be
  vacuous if no instance were rank-deficient.
- **Mutation — the prior operator must fail:** HardNet-Aff violates on **57 / 705** cases
  (rate **0.0809**, max violation **17.67**) on the identical instances.
- Cardinality clause: the enumeration size equals `Σ_{k=0..min(m,n_out)} C(m,k)` and
  `kmax == min(m, n_out)` on every cell of a 6 × 5 grid.
- **Mutation — truncate to `min(m,n_out) − 1`:** feasibility failures appear on
  **114 / 450** cases (**25.33%**), so the cardinality bound is load-bearing, not decoration.

Verdict: **verified**.

## Claim 4 — piecewise benchmark (73.33% MSE reduction, zero violations)

Route: split the claim (skill §2g).

- **Structural half — verified.** Over 1200 test points (3 seeds × 400) the CAffine model
  records **0** violations (max 4.4e-16) while the soft-penalty baseline, trained
  identically, records **413**. This is the mutation: removing the layer makes violations
  appear.
- **Magnitude half — toy.** Measured MSE 7.3826e-04 (soft) vs 6.7017e-04 (CAffine), a
  reduction of **9.22%**, far from the paper's 73.33%. `scale_note`: PAPER = CAffNet-TF
  transformer on the (unreleased) piecewise benchmark, GPU; THIS RUN = 1-32-32-2 numpy MLP,
  1500 Adam steps, 3 seeds, CPU. The reduced-scale number is **not** presented as
  reproducing the table.
- **Paper-internal arithmetic:** 1 − 0.0012/0.0045 = **73.333%**, so the quoted percentage
  is at least internally consistent with a 4:15 MSE ratio.

Verdict: **verified (structural) / toy (magnitude)**.

## Claim 5 — safety-critical control

Route: split (skill §2g); the qualitative avoid/collide outcome is scale-free, the paper's
costs and trained policy are not. Single-integrator dynamics, CBF safety constraint with a
duplicated (rank-deficient) row plus an actuator box, 12 initial states × 3 seeds × 200
steps × 3 regimes; the constraint binds the **total** command (P27).

| Regime | CAffNet | HardNet-Aff | Soft |
|---|---|---|---|
| single obstacle, loose box | 0 coll., viol 7.1e-15 | 0 coll., viol 7.1e-15 | 0 coll., viol **1.364** (and never reaches the goal: final distance 4.76) |
| two obstacles, tight box | 0 coll., viol 2.9e-15 | 0 coll., viol **0.620** | 0 coll., viol 0.0, final goal distance 2.21 |
| narrow gap, weak barrier | 0 coll., viol 2.2e-16 | 0 coll., viol 2.2e-16 | 0 coll., viol **0.252** |

**Failed prediction, recorded verbatim rather than rewritten:** the pre-registered
prediction was that at least one baseline would collide. None did, in any of the three
regimes tried. What the baselines *do* lose is the hard-constraint guarantee itself.

Sub-verdicts: `caffnet_satisfies_hard_constraint_and_avoids = verified`,
`baselines_lose_the_hard_guarantee = verified`, `baselines_actually_collide = inconclusive`.

Verdict: **partial: verified (CAffNet avoids and stays exactly feasible; baselines violate
the hard constraint) / inconclusive (no baseline collision observed)**.

## Evidence boundary

What this evidence does **not** cover:

1. **No trained CAffNet-TF.** Claims 1–3 and 5 use the layer/operator directly; claim 4 uses
   a 1-32-32-2 numpy MLP. The paper's transformer variant is untested here.
2. **No GPU, no paper-scale training.** 1500 Adam steps vs the paper's GPU-scale schedule.
   Every reported magnitude (the 9.22% reduction, the control costs) is a reduced-scale
   surrogate, not a reproduction of the tables.
3. **No paper data.** The piecewise benchmark and the control task are independent
   reimplementations from the claim text; the paper's own datasets are not released in the
   claim bundle. A different benchmark instantiation could move every magnitude.
4. **No official code executed**, so the reimplementation has no cross-check against the
   authors' implementation. Section 3's equations are the only source.
5. **The `w_scales` safeguard.** When no feasible candidate exists at full null-space scale
   the implementation shrinks `w` (1 → 0.5 → 0.25 → 0). With `w = 0` the enumeration is
   exactly the active-set enumeration of the Euclidean projection, so feasibility is
   guaranteed whenever the polyhedron is non-empty; whether the paper's layer uses the same
   safeguard is not determinable from the claim text.
6. **Dimensions and cardinalities reached** are small: `n_out ≤ 6`, `m ≤ 6`. The cardinality
   clause was checked on a 6 × 5 grid only.
7. **Claim 1's bound is an upper bound**, and the measured ratio (1.71) sits far below it; a
   tighter constant is not tested, and a *failure* to reach the bound is not evidence about
   its sharpness.
8. **Claim 5's collision contrast is untested** (see above), and the dynamics are a
   single-integrator simplification; the paper's system is not specified in the claim text.
9. **Monte-Carlo resolution.** Claim 4 uses 3 seeds; the 9.22% figure has no reported
   standard error and should not be compared to 73.33% as if it were a measurement of the
   same quantity.
10. **float64 only.** Machine-precision quantities (0.0, 1e-16) are float64 artefacts, not
    exact algebra.
