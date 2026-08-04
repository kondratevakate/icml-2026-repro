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
