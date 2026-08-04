# plan.md — arm2 reproduction of CAffNet (OpenReview 20hdQQQrA4)

Agent: Hermes (arm2, K-Dense skill set). Method skill: `paper-claim-reproduction`.

**Self-contamination note.** The `paper-claim-reproduction` skill ships a reference
`references/20hdQQQrA4-caffnet-verification.md`, i.e. a prior reproduction logbook for
*this very paper*. Per §0 of the skill it was **deliberately not opened**. Only the
skill body (generic method) and `input_bundle.json` were used. No other arm's artifacts
in this repository were read.

**Dependency cap.** numpy + scipy only, CPU (8 vCPU, no GPU). No torch, no released
official code executed. Every script is an independent reimplementation of the paper's
Section 3 equations (`caff_core.py`).

**Budget rule (explicit, from P30/P36/P54).** Fixed modest settings; no exhaustive
sweeps. Claim 1 capped at 200 trials x 3 seeds per configuration. Logbook + gate are
written as soon as the first results land, not at the end.

| Claim | Source | Route (skill §2) | Setting | Verdict ceiling |
|---|---|---|---|---|
| 1 | Theorem 3.5 (UAT + error bound `(3+3√n_out)K`) | §2l structural guarantee — sample the layer's inputs, no training | 200 trials x 3 seeds x 4 (n_out, m) cells, both branches (P25) | verified |
| 2 | Section 3, null-space term `w_phi` | §2f/P26 four-property decomposition (invariance / reachability / degenerate / benefit) | equality-geometry, machine precision | verified |
| 3 | Section 3, no full-row-rank + cardinality `min(m,n_out)` | §2l enumeration over rank-deficient dictionary + HardNet baseline must fail | deterministic dictionary + random sweep | verified |
| 4 | Experiments, piecewise benchmark, 73.33% MSE reduction, zero violations | §2g split: structural half executable, magnitude half not | numpy MLP, 1500 Adam steps, 3 seeds, CPU | structural verified / magnitude toy |
| 5 | Experiments, safety-critical control obstacle avoidance | §2g split, qualitative outcome is scale-free | 2-D control sim, 3 methods, 12 initial states, 3 seeds | verified (qualitative) |

Mutation tests planned: (1) drop the enumeration and use a single pseudo-inverse
(HardNet-Aff) on the identical instances; (2) `w = 0` must recover the fixed orthogonal
projection exactly; (3) truncate the enumeration to `min(m,n_out) - 1` — failures must
appear; (4) remove the CAffine layer from the trained model (soft penalty only) —
violations must appear; (5) same, in the control loop — the obstacle must be hit.
