# plan.md — reproduction plan (arXiv 2603.11919, ADMM on multi-affine quadratic equality constraints)

Environment: `.venv` (Python 3.12, numpy/scipy/sympy), CPU only. No official code exists
(no repository referenced in the paper), so everything is implemented from the paper's
equations (Algorithm 1, eq. 1-3, eq. 6, Section 5 toy problem).

Common infrastructure: `admm_lib.py` — exact-block ADMM (Algorithm 1) for
(a) the Section-5 toy problem, (b) generic multi-affine quadratic problems with box (polyhedral)
indicators, (c) the eq.-6 2D locomotion reduction. All blocks are solved in closed form
(each subproblem is a 1-D / small strongly convex quadratic, so "exact" as the theory requires).

Rate classification tool: fit log10(L^k - L^inf) vs k on the clean tail; a claim of LINEAR
convergence requires a straight line in that plot (constant per-iteration decay factor c^-1<1)
whereas o(1/k) sublinear behaviour shows up as log10 gap vs log10 k being straight with slope
<= -1 and the per-iteration factor tending to 1. Both diagnostics are recorded numerically.

| # | Claim | Type | CPU feasible? | Plan |
|---|-------|------|---------------|------|
| 1 | Thm 3.1 sublinear o(1/k) to a stationary point under Asm 2.3+2.6, rho large | theory + simulation | YES | `verify_claim1.py`: run Algorithm 1 on the toy problem and on Example 2.2 over an exhaustive grid of initialisations/seeds; check (a) convergence, (b) gap*k -> 0 (o(1/k)), (c) limit point is blockwise optimal & feasible. Mutation A: violate Asm 2.6 via Example 2.8 (xy=1, no z / rank-deficient Q) -> expect (x,y)->(0,0), w->-inf, infeasible. Mutation B: rho below the Thm-3.1 threshold -> expect loss of the guarantee. |
| 2 | Thm 3.2 linear rate when \|\|C\|\| small relative to Q (eq. 4) + 2nd-order diff. | simulation | YES | `verify_claim2.py`: fix C, sweep the linear coefficient q (equivalently sweep \|\|C\|\|/\|\|(QQ^T)^{-1}Q\|\|^{-1}); measure the asymptotic per-iteration contraction factor of L^k - L*, exhaustively over a seed/initialisation grid. Verified if a small-\|\|C\|\| regime gives factor bounded away from 1 and a monotone degradation as \|\|C\|\| grows. Mutation: set \|\|C\|\| large (q small) -> linear rate must be lost. Also \|\|C\|\|=0 sanity case (pure linear constraint). |
| 3 | Thm 3.3 linear rate with polyhedral indicators, no 2nd-order differentiability | simulation | YES | `verify_claim3.py`: same problem + box constraints (polyhedral) chosen so the limit is on an ACTIVE face (Lagrangian nondifferentiable there, so Thm 3.2 does not apply). Measure the Thm-3.3 quantity L^k - min_{B(x^k,z^k;r)} L(.,.,w^k) and check geometric decay. Mutation A: violate eq. 4 (large \|\|C\|\|) -> lose linear rate. Mutation B: replace the polyhedral set by a Euclidean ball (non-polyhedral) -> Thm 3.3's hypothesis broken, report what happens. |
| 4 | Cor 4.2: locomotion, linear convergence for dt <= t0 because nonlinear terms are O(dt^3) | symbolic + simulation | YES | `verify_claim4.py`: (a) SYMBOLIC (sympy) derivation of eq. 6 from eq. 5 for the 2D case, extracting the C_i coefficients and confirming their entries scale exactly as (dt)^3 while linear terms scale as (dt)^1; fit exponent of \|\|C\|\| vs dt. (b) Run ADMM on the 2D locomotion problem (m=2 kg, costs from Fig. 5) for a sweep of dt and record the contraction factor; expect linear for small dt and degradation for large dt. Mutation: artificially force the nonlinear coefficient to be dt-independent -> the dt-threshold behaviour must disappear. |
| 5 | Empirical: ADMM beats PADMM/IPDS-ADMM/IADMM; toy q>=10 gives sublinear->linear transition | simulation (part) + underspecified benchmark (part) | PARTIAL | `verify_claim5.py`: the q>=10 half is directly testable on the Section-5 toy problem — exhaustive q sweep, measure the contraction factor and locate the transition. The baseline-comparison half (Fig. 4) has NO released code, NO hyperparameters (proximal/step-size parameters of PADMM, IPDS-ADMM, IADMM), NO problem instances and NO seeds in the paper; any re-implementation would be my own choice of baselines, not theirs. That half is reported `inconclusive` with the reason, not faked. |
| 6 | Validation on 2D locomotion sim + real robot (humanoid jump, quadruped bounding) | hardware | NO | STOP. Real-robot experiments (Figure 6) require a physical humanoid and quadruped, plus the unreleased kinematics/DDP pipeline and logs. Not reproducible on CPU or at all from the released material. Verdict `inconclusive` (hardware unavailable). The 2D-simulation part is covered by claim 4's evidence and is reported there. |

Budget tracking: start 2026-08-02 00:30 local. Priorities: claims 1-3 (cheap), 4, 5, then 6 (refusal).
