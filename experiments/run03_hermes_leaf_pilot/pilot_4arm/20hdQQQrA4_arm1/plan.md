# plan.md — CAffNet (orid 20hdQQQrA4), arm1 control run

| # | Claim | Type | CPU feasible? | Approach |
|---|-------|------|---------------|----------|
| 1 | Thm 3.5 bound (3+3√n_out)K | theory | yes | Construct instances satisfying the proof's hypotheses exactly (f_t feasible, ‖f_θ−f_t‖<K, ‖w_φ‖<2K); measure realised ‖P*−f_t‖ against the bound; also check the Eq. 31 matrix-norm bounds and the intermediate (1+3√n)K bound at the intersection γ. Exhaustive over n_out∈{1..5}, m∈{1..7}, p∈{1,2,3}, rank regimes × seeds. Mutation: break Eq. 12 (argmax instead of argmin). |
| 2 | Eq. 8 trainable null-space w_φ | theory + small sim | yes | (a) A_γ P_γ = b_γ for arbitrary w_φ; (b) output varies with w_φ iff A_γ is rank-deficient; (c) fitting w_φ beats the fixed orthogonal projection on a face-selection task. Mutation: freeze w_φ = 0. |
| 3 | No full-row-rank; cardinality ≤ min(m,n_out) | theory | yes | Feasibility sweep on rank-deficient / redundant / m≫n_out instances; HardNet-style single-pseudoinverse baseline on identical instances; truncated Γ (k=1 only, k=min only) as mutations; Fig. 1 example. |
| 4 | 73.33% MSE reduction, zero violations (§4.1) | training | yes (torch CPU) | Reimplement App. D.1 target + 4 piecewise constraints; train soft NN (FF 3×200) and CAffNet-TF (3 heads×40, ff 120), Adam 1e-4, 50000 epochs, 5 seeds, 50 train / 400 test points. Mutation: same transformer with the CAffine layer removed. |
| 5 | Safety-critical control obstacle avoidance (§4.3) | training + sim | **partially — spec incomplete** | The paper gives obstacles, boxes, PID gains, x0, but not the reference trajectory, dt, horizon, training loss, or the affine per-step safety constraint (obstacle avoidance is non-convex). The trained policies of Table 4 are therefore NOT reproducible from the paper alone → the headline claim gets `inconclusive`. A well-posed mechanism-level sub-test is run instead: identical PID nominal + identical CBF-style affine constraints (m=7 > n_out=2), CAffNet projection vs soft vs HardNet-style. |

Budget: soft target 4h. Claims 1–3 and 5 are pure numpy; claim 4 is the only long job (15 runs × 50000 epochs, parallelised 1-thread-per-process over 6 cores).
