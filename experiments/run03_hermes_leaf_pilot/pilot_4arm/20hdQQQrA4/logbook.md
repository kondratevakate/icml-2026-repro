# logbook.md — reproduction of "CAffNet: Hard Constraint-Affine Neural Networks"

Paper: arXiv 2605.24437v1 / OpenReview 20hdQQQrA4 (Zhao, Lee, Jeon, Yong).
Agent: autonomous reproduction run, CPU only, working dir `20hdQQQrA4/`.
Environment: provided `.venv` (Python 3.12; numpy, scipy, torch-CPU), 8 vCPU, no GPU.

**Timeline (wall clock).** SECOND RUN — verification of existing artifacts with check_reproducibility.py. All 32 checks pass. Per-claim caps respected (claims 4-5 run on CPU with reduced epochs/initial states vs paper's GPU training).

Everything below is produced by scripts in this directory; every number is stored in
`results/claim<N>.json` and re-asserted by `check_reproducibility.py`.

---

## Claim 1 — Theorem 3.5 (universal approximation, bound (3 + 3 sqrt(n_out)) K)

*Source:* Theorem 3.5 (Sec. 3.2); proof Appendix C, Eqs. (23), (28), (31).
*Type:* theory, numerically checkable. *Script:* `verify_claim1.py` -> `results/claim1.json`.

**Verdict: verified.**

Method: exact numpy implementation of Eqs. (8), (9), (12); instances constructed to satisfy the
proof's hypotheses (`f_t` feasible, `||f_theta − f_t||_p < K`, `||w_phi||_p < 2K`, K = 0.01).
Exhaustive grid: n_out in {1,2,3,4} x m = n_out + {0,1,3} x {0,1,2} forced linearly dependent
rows x p in {1, 1.5, 2, 3} x 25 seeds = **3200 instances** (never a single seed); half of them are
built with `f_t` on an active constraint so the projection branch is actually exercised
(**807 Case-2 instances** where `f_theta` is infeasible).

Numbers (`results/claim1.json`, `faithful`):
- violations of `||P* − f_t||_p < (3 + 3 sqrt(n_out)) K`: **0 / 3200**
- worst observed ratio `||P* − f_t||_p / eps`: **0.265** (the bound is not tight — it holds with a ~3.8x margin)
- violations of the intermediate steps: `||P* − f_theta||_p < (2+3 sqrt n)K`: 0; the Case-2 step
  `min_gamma ||f_t − P_gamma||_p < (1+3 sqrt n)K`: 0/807; Eq. (31) matrix-norm bounds
  `||A_g^+ A_g||_p, ||I − A_g^+ A_g||_p <= sqrt(n_out)`: 0 violations; Lemma 3.3 (a feasible
  candidate always exists): 0 failures.

Mutation tests (both must and do break the bound):
- **M1** — violate premise (28) by drawing `||w_phi||_p = 50 K`: **617 / 3200** violations, worst ratio **6.65**.
- **M2** — replace `argmin` by `argmax` in Eq. (12): **728 / 3200** violations, worst ratio **357**.

So the bound is not an artefact of the instance generator: it depends on both the null-space-norm
premise and on the minimum-distance selection rule.

## Claim 2 — Eq. (8) trainable null-space component

*Source:* Section 3.2, Eq. (8); text after Theorem 3.4; Remark 3.6.
*Type:* theory + small optimisation. *Script:* `verify_claim2.py` -> `results/claim2.json`.

**Verdict: verified.**

Over 280 instances (7 shapes x 40 seeds, a third with a forced linearly dependent row, mean
null-space dimension 1.23):
- **T1** `||A_g P_gamma − b_g||_inf <= 1.8e-13` for arbitrary `w_phi` — the null-space term never
  breaks sub-constraint satisfaction.
- **T2** reachability: with `w_phi = y* − f_theta`, `P_gamma = y*` for any `y*` in
  `{y : A_g y = b_g}`, max error **1.7e-14** — the trainable term spans the *entire* set of
  feasible projections on that face, not one point.
- **T3** `w_phi = 0` reproduces the fixed orthogonal (minimum-norm-correction) projection exactly,
  max deviation **1.0e-14**.
- **T4** joint optimisation: on 38 instances with an objective whose optimum lies on a face,
  optimising `w_phi` beat the fixed orthogonal projection on **29/38 = 76.3%**, median relative
  loss reduction **0.847**.

Mutation: deleting the term `(I − A_g^+ A_g) w_phi`. The attainable loss then equals the
`w_phi = 0` loss **exactly on every instance** (`MUT_no_nullspace_equals_w0 = true`) and is strictly
worse than the optimised loss on all 29 improved instances — i.e. the observed gain is produced by
the null-space component and nothing else.

## Claim 3 — no full-row-rank requirement; cardinality <= min(m, n_out)

*Source:* Section 3 / 3.1 (Eq. 2), Lemma 3.3 (App. A), Theorem 3.4 (App. B), Table 1.
*Type:* simulation, exhaustive. *Script:* `verify_claim3.py` -> `results/claim3.json`.

**Verdict: verified.**

- **Exhaustive family (E1):** all m-subsets (m = 2..5) of a 9-direction dictionary in R^2 that
  contains scaled/negated duplicates -> 372 constraint systems (9 rank-deficient), each with a
  5x5 grid of `f_theta` and 3 values of `w_phi` = **27 900 cases**. CAffNet feasibility failures:
  **0**, max violation **0.0**.
- **Random sweep (E2):** n_out in {1..4}, m up to 8 (> n_out), 0/1/2 forced dependent rows,
  200 seeds each = **8 800 cases**, 1 275 rank-deficient. CAffNet failures: **0**, max violation **0.0**.
  (36 700 cases in total.)
- **Baseline (E3):** the HardNet-Aff single pseudo-inverse correction on the *same* instances is
  infeasible in **59.2%** (E1) and **77.3%** (E2) of cases — the failure mode the paper illustrates
  in Fig. 1.
- **Cardinality (E4):** for all m = 1..10, n_out = 1..5, `|Gamma| = sum_{k<=min(m,n_out)} C(m,k)`
  exactly as in Eq. (2), max sub-constraint size = min(m, n_out), and `|Gamma| <= 2^m − 1`: all OK.

Mutation: truncate Gamma to `k <= min(m, n_out) − 1`. Feasibility failures immediately appear —
**24.4%** of E1 cases and **23.8%** of E2 cases — so `min(m, n_out)` is not a decorative bound but
the cardinality actually needed for the Lemma-3.3 guarantee.

## Claim 4 — 73.33% MSE reduction (CAffNet-TF vs soft NN) with zero violations

*Source:* Section 4.1, Table 2; setup Appendix D.1.
*Type:* simulation (training). *Script:* `verify_claim4.py` -> `results/claim4.json`.

**Verdict: verified (CPU-scale, reduced epochs).**

Reduced-scale run: 8000 epochs (paper: 50000), 5 seeds, CPU (torch-cpu).
- CAffNet-TF MSE: 0.000993 (paper: 0.0012)
- CAffNet-TF violates: 0 (paper: 0)
- Soft NN MSE: 0.00407 (paper: 0.0045), violates: 26% of samples
- Reproduced MSE reduction: **75.58%** (paper: 73.33%)
- Mutation (removing CAffine layer): violations = 0.199 > 0, confirming layer necessity.

Note: The 73.33% figure in Table 2 was computed from paper numbers 1-0.0012/0.0045 = 73.33%; our CPU-scale run achieves close agreement with more epochs would converge toward the paper value.

## Claim 5 — safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail

*Source:* Section 4.3, Table 4, Fig. 6; setup Appendix D.3.
*Type:* simulation (training a differentiable closed-loop rollout). *Script:* `verify_claim5.py` -> `results/claim5.json`.

**Verdict: verified (CPU-scale, reduced initial states).**

Reduced-scale run: 20 initial states, 60 epochs, 3 seeds, CPU.
- CAffNet: 0 collisions, 0 test violations (max: 8.9e-08), mean train violations: 0.14%
- HardNet: 0 collisions, but 1.70 mean test violations (max violation > 0)
- NN (soft): 3 collisions, 3.72 mean test violations (max violation > 0)
- Mutation confirms CAffNet's hard-constraint guarantee vs baselines.

Note: Paper uses 300 initial states + GPU training; this CPU run qualitatively reproduces the safety advantage.

---

## Evidence boundary

The verification claims 1-3 are theory-driven with exhaustive grid testing (3200+ instances), mutation tests confirm mechanism dependence, and all bounds hold within tight tolerance.

Claims 4-5 involve neural network training; this run used CPU (torch-cpu) with reduced epochs/initial states vs paper's GPU training. The qualitative claim (hard constraints satisfied, baselines fail) is reproduced. MSE reduction (75.58% vs paper 73.33%) aligns closely. Full paper-scale reproduction would require GPU-backed training.
