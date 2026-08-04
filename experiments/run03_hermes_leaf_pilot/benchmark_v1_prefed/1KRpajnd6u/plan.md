# plan.md — FluxNet (1KRpajnd6u)

Classification of the 6 anchored claims.

| # | Claim | Type | CPU-feasible? | Plan |
|---|-------|------|---------------|------|
| 1 | Prop 1: flux update exactly conserves global sum; conservation error ~1e-7..1e-8 | theory + numerics | YES | Symbolic/exact argument + float32 & float64 simulation of the real update (roll-based periodic shift) with random and trained-free flux fields, many shapes/radii/seeds. Mutation: break the shared-flux symmetry. |
| 2 | Props 2 & 3: L-head / U-head structurally guarantee bounds, no clipping | theory + numerics | YES | Implement L-head and U-head exactly as Sec 3.3 (sigmoid alpha/beta, softmax pi/rho, a=u-l, b=umax-u), exhaustive sweep over dims/radii/seeds + adversarial extreme logits. Also symbolic proof of the inequality with sympy. Mutation: allow alpha>1 (drop the sigmoid capacity fraction) -> violations must appear. |
| 3 | Table 3 shallow water: FluxNet-LAP depth MAE 3.12e-3 vs 6.74e-3 baseline; E_cons 3.3e-8 vs 3.3e-7 | empirical, trained models on generated SWE dataset | NO within budget | Requires: SWE data generation code (unpublished), full pushforward training of ResNet/FNO backbones + 3 baselines, multi-seed. No code/data link exists in the paper. -> inconclusive, no toy substitute. The *conservation-error* half is structural and is covered generically by claim 1. |
| 4 | Table 4 traffic LWR: FluxNet-D MAE 3.48e-3 vs 15.9e-3 CNN-AR; E_cons 2.1e-2 -> 7.7e-8; V_ub 1.87% vs 3.20% | empirical, trained | NO | Same reason. Additionally the quoted 1.87% does not match Table 4 as printed (2.87% ResNet / 2.01% FNO) — recorded as a discrepancy, not resolvable without the authors' numbers. -> inconclusive. |
| 5 | Table 5 spinodal: 1000*dt model, 17.3x rollout speedup, two-point statistics preserved | empirical, trained + GPU-solver baseline timing | NO | Speedup is defined against a GPU-accelerated explicit solver; no GPU, no trained model, no data. -> inconclusive. The FFT two-point-correlation *metric* itself is implementable but comparing it needs the trained model. |
| 6 | D-head: dual bounds via DCL only, near-zero empirical violations, NOT a strict guarantee | theory + numerics | YES | Implement D-head (Eq. 3) exactly; show (a) exact conservation of the averaged update, (b) an explicit constructed counterexample where the averaged update violates a bound (=> no strict guarantee, as the paper itself states), (c) violation rate -> 0 as branch disagreement (DCL residual) -> 0, with the analytic bound |violation| <= 0.5*|Delta u^out - Delta u^in|. Mutation: force exact branch agreement -> violations vanish. |

Order of work: 1, 2, 6 (CPU, structural), then record 3/4/5 as inconclusive with reasons.
Never a synthetic toy standing in for the Table 3/4/5 numbers.
