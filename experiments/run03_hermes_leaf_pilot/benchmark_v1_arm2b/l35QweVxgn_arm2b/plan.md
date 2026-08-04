# plan.md — reproduction plan, orid l35QweVxgn (arm2)

Paper: *On the Theory of Continual Learning with Gradient Descent for Neural Networks*,
arXiv 2510.05573 (OpenReview l35QweVxgn). Area: Theory.
Arm: **arm2** (Hermes + K-Dense skill set, no context compaction). CPU-only, numpy/scipy/sympy.

Skill discipline (SKILL.md §0): this paper is NOT one of the papers covered by the skill's
paper-specific references, so no reference logbook was opened for it. Generic method
(SKILL.md body + `references/theory_only_paper_repro.md`) only. No other arm's logbook read.

## Nature of the paper
Pure theory (4 theorems + propositions) with a small empirical section. A code repo exists
(github.com/hosseinta2/continual-learning-with-neural-nets.git) but it is *not* used here:
all verification is an **independent numerical re-derivation from the paper's own equations**
(Eq. 1–6, Eq. 17, Thm 2.1/2.2/2.3/B.1). That changes what `verified` means — the theorem's
conclusion held on *our* instantiation of its hypotheses.

## Model / DGP as instantiated
- Two-layer net, quadratic activation (App. C), `Φ(w,x) = (1/√m) Σ_i a_i ⟨w_i,x⟩²`,
  `a_i = ±1` fixed, `w_i^(0) ~ N(0, I_d)`, only first layer trained, full-batch GD, hinge loss.
- Task `k`: XOR clusters (Eq. 2) with `μ^k_+ ⊥ μ^k_-`, cross-task means orthogonal too,
  noise `σ = Θ(1/(log^c d · √d))`.
- **Ambiguity (P44)**: the PDF renders `‖μ‖ = Θ(1/√d)` but with `σ√d = 1/log^c d` that makes
  noise dominate the signal, contradicting the XOR-cluster SNR literature the paper cites.
  Both readings (`‖μ‖ = 1/√d` literal, `‖μ‖ = 1` SNR) are swept where it matters.

## Routing (SKILL.md §2)
| Claim | Content | Route | Ceiling |
|---|---|---|---|
| 1 | Thm 2.1 closed-form order of `F^tr_{k,K}` | exponent recovery on the paper's own Eq. 17 characterisation (analytic + MC) | verified |
| 2 | regime `n=Θ̃(d²K)`, `m=Ω̃(d⁸K⁴)`, `ηT=Θ(d²)` gives `F^tr=o_d(1)` | simulation on a `d`-ladder; `m=d⁸K⁴` is astronomically infeasible on CPU | **toy** (scaled width), sub-verdicts |
| 3 | Thm 2.2 uniform small train error over K tasks | simulation (real GD) | verified |
| 4 | Thm 2.3 gap `≲ ηT e^{ηT(K-k+1)/√m}/n`, decays in n | simulation + bound check | verified |
| 5 | Thm B.1 polylog-in-T vs linear-in-T | measured loss trajectory + closed-form bound comparison + sympy | verified |
| 6 | decomposition `F^ts ≤ F^tr + F^gen`; width & n jointly | identity check + 2-D (m,n) sweep | verified |

Every `verified` gets a mutation test with a pre-registered predicted direction.
Seeds fixed in `common.py`; MC standard errors reported.
