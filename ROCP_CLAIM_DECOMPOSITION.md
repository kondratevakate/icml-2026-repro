# ROCP Claim Decomposition

Paper: *Optimal Decision-Making Based on Prediction Sets* (`VAXW59dyfk`)
Source checked: arXiv `2602.00989v3`, official repository, and the local
anchored-claim snapshot. Date: 2026-07-28.

## Decision

The defensible score path is **4 points now, 6 only after an independent
paper-faithful conformal implementation**. The official repository is valuable
for the cached empirical inputs, but it is not a direct implementation of
Algorithm 1's test-point-inclusive calibration rule.

| Claim | Paper anchor | Full verdict route | Do not claim | Target |
|---|---|---|---|---:|
| C1: minimax policy for fixed prediction set | Lemma 2.1, Theorem 2.2, Eqs. (3)-(5) | Derive the closed-form adversarial risk and policy, then exhaustively compare it against the primal finite-label linear program over nonempty sets, actions, loss matrices, and alpha values. Include a counterfactual max-min-only policy showing the out-of-set penalty changes the selected action. | A few random numerical examples without the derivation. | 2 |
| C2: optimal prediction-set construction | Proposition 3.1, Theorem 3.3, Eqs. (9)-(13) | Not a throughput target. A full audit must cover the non-atomic-space assumption, normal-integrand condition, Fenchel-Rockafellar duality, randomized-kernel relaxation, and derandomization. | Discrete-grid simulation as proof of Theorem 3.3. It violates the theorem's non-atomic setting and is at most toy evidence. | 0 |
| C3: ROCP and finite-sample coverage | Algorithm 1, Section 4.2 | Implement Algorithm 1 exactly: candidate-label calibration includes the candidate test label in the `(n+1)` coverage condition. Exhaustively audit small exchangeable finite problems and run empirical coverage on all cached COVID seeds and BDD splits. Document that the paper only *expects* oracle-risk approximation under consistency/stability; verify the construction and coverage, not a nonexistent finite-sample optimal-risk theorem. | Treating the current official `rocp.py` as Algorithm 1 without audit. Its `calibrate_beta()` explicitly omits the test point and uses one global beta. | 2 |
| C4: medical and safety-critical empirical gain | Fig. 1, Fig. 2, Section 5 | Run COVID Lambda0 and Lambda1 over cached seeds 23-42, all seven alpha values, with ROCP/RAC/LAS/APS/SOCOP/best-response. Run BDD on cached 10,000 hazard records over 20 fresh splits. Recompute all four metrics and verify the direction plus the larger Lambda1 gain. | One COVID seed, only Lambda0, or a synthetic proxy. Those are toy scope. | 2 |

## Code Divergence That Must Be Addressed

The paper's Algorithm 1 calibrates each candidate label with the calibration
sample **and** the candidate test label in the `(n+1)` coverage constraint. The
repository's `rocp.py` instead documents that `calibrate_beta()` uses only
calibration empirical coverage and "does NOT include the test point". A second
divergence occurs at `t=0`: Remark 3.2 requires `C(x,0)=Y`, while the released
code uses the minimum action loss as its threshold and can return a strict
subset of `Y`. These are implementation divergences, not yet a falsification
of the paper's theorem.

Therefore:

1. Do not use the existing evaluator to declare C3 `verified`.
2. Do not declare C3 `falsified` merely because the repository differs from the paper.
3. Add a small independent Algorithm 1 implementation and an exchangeability
   audit before assigning any C3 verdict.

## Local Execution Status

The independent bundle is now in `rocp/repro_rocp`:

- C1 passed 28,950 action-value comparisons against independently solved
  primal linear programs; maximum absolute error was `1.42e-14`, with no
  minimizer-set failures.
- C3 passed 14,850 exhaustive exchangeability-orbit/alpha checks over a
  nine-observation support through sample size `n+1=6`.
- An executable comparison against official commit
  `3ee0cf6e393d2e434368bc1fc7fe3abd03ed493f` reproduced both code divergences
  and found concrete cases where released global-beta sets differ from the
  paper-faithful candidate-wise sets.
- C3 is now verified after the finite audit and full cached coverage check.
- C4 is verified from 20 COVID seeds under both loss matrices and 20 BDD
  splits. The anchored critical-mistake direction passes; a stronger BDD
  all-baselines/all-alpha statement has small RAC reversals.

## Preflight Acceptance Criteria

### C1

- Formula from Eq. (3) equals the primal adversarial optimum for every finite
  enumeration case.
- The Eq. (4) action equals the primal minimax policy.
- At least one asymmetric-loss case separates ROCP from the alpha-zero max-min rule.

### C3

- For every enumerated exchangeable case, the paper-faithful `(n+1)` procedure
  meets the exact marginal-coverage bound.
- Cached COVID and BDD experiments report empirical miscoverage for every alpha
  and clearly distinguish this empirical check from the distribution-free proof.
- The reproduction implementation is compared directly with the repository,
  with the test-point calibration difference named explicitly.

### C4

- COVID: 20 cached seeds, both Lambda0 and Lambda1, seven alpha values, all
  paper baselines, and critical-mistake rates at alpha=0.05.
- BDD: 20 fresh deterministic splits from `results/BDD/raw_bits.npz`; all four
  reported metrics and collision-penalty critical mistakes.
- Report exact metric deltas versus RAC and best-response, not just a plot.

## Execution Sequence

1. Implement and unit-test the C1 primal-vs-closed-form audit.
2. Implement the paper-faithful C3 calibrator and finite exchangeability audit.
3. Run C4 from the official cached inputs.
4. Ask a fresh judge-preflight model to attack every claimed `verified` or
   `falsified` verdict before creating the logbook.

## Updated Estimate

- C1 + C3 + C4: completed, forecast `6/8`.
- C2 is excluded from the throughput forecast.
