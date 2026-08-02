# logbook.md — reproduction of arXiv 2602.01603 (OpenReview Km8yqb7SfN)
**"Inference-Aware Meta-Alignment of LLMs via Non-Linear GRPO"**

Elapsed: first tool call 2026-08-01 23:57 (+04) → finish 2026-08-02 00:56 (+04) ≈ **59 min**
(soft target 4h, hard stop 8h — well inside budget; no claim consumed more than ~25 min).
LLM turns used: ~22 for the whole paper (budget 60). Paper read **once** → `notes_paper.md`.
No official code repository URL exists in the paper HTML (code is in unavailable supplementary
material), so `notes_code.md` was not created — nothing was read from any code repo.

Everything below is CPU-only (numpy/scipy/sympy), reproduced on a **finite response space**
(the paper's own Section 3.1 setting is 1-D, and Section 4.1's BoN derivative only depends on the
1-D reward distribution), plus the continuous [0,1] setting for Proposition 3.1.

Shared substrate (`iama_core.py`): exact discrete BoN objective
`R[p] = Σ_k r_k (F_k^N − F_{k−1}^N)`, its exact functional derivative
`dR/dp_j = −Σ_{k≥j} (r_{k+1}−r_k) N F_k^{N−1}` (the discrete form of Prop. 4.2), the closed-form
exact proximal step of Algorithm 1, and an **independent** optimum solver (L-BFGS-B on a softmax
parametrization) used as ground truth so that the algorithm under test is never its own reference.

---

## Summary table

| claim | verdict | one-line evidence |
|---|---|---|
| 1 — Eq. (1) objective; BoN makes R non-linear in π; one base policy adapts to m criteria (Sec. 3) | **verified** | chord-vs-curve gap of R under BoN is 1.0e-4…8.1e-2 over 45 instances while the untransformed E_π[r] gap is 2.2e-16; IAMA-trained base beats the naive base on the IAMA objective in 60/60 instances (min gain 5.7e-3). Mutation N=1: gap 3.3e-16 and the two optima coincide (TV = 0). |
| 2 — Algorithm 1 = GRPO with the reward replaced by ∂R/∂π at the empirical π̂ (Sec. 4) | **verified** | analytic ∂R/∂π matches central finite differences of Definition 4.1 to 8.3e-10 abs / 7.3e-8 rel over 80 cases; the unchanged GRPO mirror-descent map fed with r̃ reaches the independent optimum (suboptimality 5.6e-16, TV 1.0e-8) in 30/30 instances; empirical-π̂ error decays as M^(−1.02). Mutation (plain reward = standard GRPO): strictly worse in 30/30, median excess loss 3.2e-2. |
| 3 — Theorem 5.2: L[π_T]−L[π*] ≤ β·KL[π*‖π_0] / (((L+β)/L)^T − 1) (Sec. 5.1) | **verified** | over 90 instances × 25 iterates (2250 checks) there is **no** violation with excess loss above 1e-12; the 44 raw "violations" all have LHS ≤ 3.3e-16 (float64 noise) at T ≥ 18; concavity of Assumption 5.1 violated 0 times. Mutations: η = 20/L → 11/36 macroscopic violations (worst excess loss 0.59); plain reward instead of ∂R/∂π → 36/36 violations, 35 macroscopic. |
| 4 — Theorem 5.3: inexact updates converge to a neighbourhood with additive bias 2(ε+δ)/β (Sec. 5.2) | **verified** | over 60 noise/β/K/N configs × 200 runs each, E[L[π_t̂]]−L[π*] never exceeds the full bound (max ratio 0.261); dropping the 2(ε+δ)/β term breaks the bound in 36/36 noisy configs (max ratio 1.65e10), and the measured floor is monotone increasing in noise in 12/12 (K,N,β) cells. ε and δ were **measured** from the runs, not assumed. |
| 5 — Proposition 3.1: naive optimum = δ_0.5, IAMA/BoN optimum π*(y) ∝ y^(α−1)(1−y)^(α−1)/(y^α+(1−y)^α)² , α = 1/(N−1) (Sec. 3.1) | **verified** | the closed form is exactly the derivative of the CDF y^α/(y^α+(1−y)^α) (sympy) and integrates to 1; at π* the aggregate functional derivative 0.5·∂R₁/∂π + 0.5·∂R₂/∂π is constant to a relative span ≤ 5.8e-16 for N=2,3,4,8,16 — the exact β=0 first-order optimality condition; grid mirror ascent matches its objective value to ≤ 1.4e-5; naive optimum puts mass 1.0 within 0.01 of y = 0.5 (std 1.25e-3) while IAMA (N=8) puts 0.922 outside [0.25,0.75]. Mutations: N=2 ⇒ exactly uniform (max |π*−1| = 0); α = 1/N ⇒ optimality span jumps to 3.1e-2…1.6e-1. |
| 6 — Sec. 6.2 HH-RLHF / Alpaca-7B: BoN(N=4) IAMA pushes the helpfulness–harmlessness Pareto front forward, no-BoN IAMA ≈ standard alignment | **inconclusive** | needs multi-run RL training of a 7B policy plus two unreleased Qwen3-4B Bradley-Terry reward models and Qwen-32B golden judges; no public code URL in the paper; the paper's evidence is a figure with no numeric table. Refused rather than substituted with a toy. |

---

## Claim 1 — Equation (1), Section 3 → **verified**
Script `verify_claim1.py`, data `results/claim1.json`, command `.venv/bin/python verify_claim1.py`.

* Non-linearity (the paper's stated distinguishing property of Eq. (1)): for random policies π_a, π_b
  and t on a 19-point grid, `max_t |R[(1−t)π_a+tπ_b] − ((1−t)R[π_a]+tR[π_b])|` over 45 instances
  (K ∈ {4,6,10} × N ∈ {2,4,8} × 5 seeds) lies in **[1.037e-4, 8.139e-2]**, whereas the same measure
  for the untransformed E_π[r] is **2.22e-16** (exactly linear, as the paper states).
* Multi-criterion adaptation: 60 instances (K ∈ {6,10} × N ∈ {2,4,8} × m ∈ {2,3} × 5 seeds) with
  conflicting rewards (r₂ = 1 − r₁), β = 0.05. The base policy trained on Eq. (1) achieves a higher
  Eq.-(1) value than the transform-unaware ("naive") base in **60/60** cases, min gain **5.726e-3**,
  median **4.096e-2**.
* **Mutation** (N = 1, i.e. BoN degenerates to the identity transform, removing the mechanism):
  chord gap collapses to **3.33e-16** (linear again) and the IAMA and naive optima become identical
  (max TV = **0.0** over 10 instances) — exactly as predicted if the non-linearity is caused by T_i.
* Cross-check: mirror descent vs the independent L-BFGS solver agree to TV = 5.16e-9.

## Claim 2 — Algorithm 1 / Proposition 4.2, Section 4 → **verified**
Script `verify_claim2.py`, data `results/claim2.json`.

* Definition 4.1 check: `<∂R/∂π, π′−π>` vs central finite differences of `R[π+ε(π′−π)]`, 80 cases
  (K ∈ {4,6,10,20} × N ∈ {2,3,4,8} × 5 seeds): max abs error **8.285e-10**, max rel error **7.33e-8**.
* "Drop-in on top of standard GRPO": the standard mirror-descent/GRPO update is left untouched and
  only the reward vector is replaced by r̃ = ∂R/∂π. Over 30 instances it reaches the true optimum of
  the non-linear objective: max suboptimality **5.55e-16**, max TV to the independent optimum **1.01e-8**.
* Empirical π̂ (Algorithm 1 line 4, M sampled responses): E‖∂R/∂π[π̂] − ∂R/∂π[π]‖²_sp measured for
  M = 8…1024 (400 reps each) gives a log-log slope of **−1.024** vs M (theory: −1, the Section 5.3
  O(1/M) lemma). At M = 8 (the paper's setting) the mean squared span error is 4.02e-2.
* **Mutation** (feed the plain scalar reward, i.e. plain GRPO): the fixed point is strictly worse on
  the IAMA objective in **30/30** instances, min excess loss **5.471e-3**, median **3.168e-2**.

## Claim 3 — Theorem 5.2, Section 5.1 → **verified** (up to a numerically estimated L)
Script `verify_claim3.py`, data `results/claim3.json`.

* Grid: K ∈ {5,8} × N ∈ {2,4,8} × β ∈ {0.1,0.5,1.0} × 5 reward seeds = **90 instances**, each checked
  at every T = 1…25 (**2250** bound evaluations). π₀ = π_ref = uniform; π* and L[π*] from L-BFGS.
* L is not given in closed form in the paper's main text, so it is estimated as the smallest constant
  satisfying Assumption 5.1 over 4000 random policy pairs per instance, then inflated ×1.25.
  The concavity half of Assumption 5.1 was violated **0** times (max violation exactly 0.0).
* Result: **0** violations with excess loss above 1e-12. The 44 raw ratio>1 rows all occur at T ≥ 18
  with LHS ≤ **3.33e-16**, i.e. double-precision noise where both sides are already ≲1e-16.
  Max excess loss at T = 25 across all instances: **2.04e-6**, consistent with linear convergence.
* **Mutation A** (η = 20/L, violating the theorem's step size): 21/36 ratio violations, **11**
  macroscopic (final excess loss > 1e-3), worst final excess loss **0.593**.
* **Mutation B** (replace ∂R/∂π by the plain reward): **36/36** violations, 35 macroscopic,
  final excess loss 9.0e-4…8.9e-2 — the guarantee is specific to the functional-derivative update.

## Claim 4 — Theorem 5.3, Section 5.2 → **verified**
Script `verify_claim4.py`, data `results/claim4.json`.

* 60 configs = K ∈ {5,8} × N ∈ {2,4} × β ∈ {0.2,0.5,1.0} × 5 (noise, residual) settings
  ((0,0), (0.05,0), (0.2,0), (0.2,0.05), (0.5,0.1)); T = 20 steps; **200 independent runs each**;
  the LHS is averaged with the theorem's index distribution Prob(t̂ = t) ∝ ((L+β/2)/L)^t.
* ε is measured as the empirical mean of ‖noise‖²_sp and δ as the mean of ‖r_t‖²_sp with
  r_t = −r̃_t + β log(π_{t+1}/π_ref) + (1/η) log(π_{t+1}/π_t) computed from the actual iterates —
  no assumed values.
* Result: the full bound `(β/2)KL[π*‖π₀]/(((L+β/2)/L)^T − 1) + 2(ε+δ)/β` is **never** violated;
  max LHS/RHS ratio **0.2611** over all 60 configs.
* **Mutation** (drop the additive 2(ε+δ)/β term, i.e. test the exact Theorem-5.2 form): violated in
  **36/36** noisy configs, max ratio **1.65e10** — the bias term is necessary, not decorative.
* The measured plateau grows monotonically with the injected noise in **12/12** (K,N,β) cells.

## Claim 5 — Proposition 3.1, Section 3.1 → **verified**
Script `verify_claim5.py`, data `results/claim5.json`.

* Normalization: sympy confirms d/dy [y^α/(y^α+(1−y)^α)] equals the claimed density exactly, so the
  CDF is closed-form and F(1)−F(0) = 1; numeric quadrature gives mass 1.000, 1.002, 1.000, 1.000 for
  N = 2,4,8,16 (the 2e-3 deviation at N = 4 is quadrature error at the endpoint singularity).
* Optimality: for β = 0 an interior simplex maximiser must have a constant functional derivative.
  Using Prop. 4.2 with r₁(y) = 1−y², r₂(y) = 1−(1−y)²:
  ∂R₁/∂π(y) = −∫₀^y 2Nz(1−F(z))^{N−1}dz, ∂R₂/∂π(y) = −∫_y^1 2N(1−z)F(z)^{N−1}dz.
  At the claimed π*, the relative span of 0.5(∂R₁/∂π + ∂R₂/∂π) on [0.02, 0.98] is
  **0.0 / 4.16e-16 / 4.54e-16 / 5.84e-16 / 5.74e-16** for N = 2,3,4,8,16 — machine precision.
* Grid optimisation (K = 400 bins, exact mirror ascent, 5 inits): objective matches the closed form to
  ≤ **1.45e-5**; from the uniform init at N = 2 the TV to the closed form is 6.3e-15. From random
  inits the TV can stay as large as 0.56 while the objective gap is < 1.5e-5 — the β = 0 objective is
  extremely flat near the optimum on a 400-bin grid, so the TV agreement is weak evidence and the
  verdict rests on the exact first-order-optimality test above.
* Naive optimum: 0.5E[r₁]+0.5E[r₂] optimisation puts mass **1.0** within 0.01 of y = 0.5
  (argmax bin 0.49875, mean 0.5, std 1.25e-3) — the claimed δ_0.5 collapse — and **0.0** mass outside
  [0.25, 0.75], versus **0.922** for IAMA with N = 8 (the claimed bimodality).
* **Mutation A**: N = 2 ⇒ α = 1 ⇒ the formula must be exactly uniform; max|π*(y) − 1| = **0.0**.
* **Mutation B**: use α = 1/N instead of 1/(N−1); the optimality span jumps to **0.162, 0.122, 0.062,
  0.031** for N = 3,4,8,16 — i.e. 14–15 orders of magnitude above the correct exponent.

## Claim 6 — Section 6.2 (HH-RLHF / Alpaca-7B) → **inconclusive**
Script `verify_claim6.py`, data `results/claim6.json`. No toy substitute was produced.
Blocking reasons (all recorded in the JSON): (i) one RL run of a 7B policy per Pareto weight w₁ with
TRL GRPO and M = 8 samples/prompt — GPU-cluster scale, impossible on CPU within the 2h per-claim cap;
(ii) the two Qwen3-4B Bradley-Terry reward models and the two Qwen-32B golden judges are author-trained
and unreleased; (iii) no public code URL in the paper (supplementary only); (iv) the claim's evidence
is Fig. 6b, a figure with no numeric table, so there is no reported number to re-assert.
hh-rlhf itself is public — the blockers are the reward models, the missing code and the GPU scale,
not dataset access.

---

## Evidence boundary — what this evidence does NOT cover

1. **No LLM was trained or run.** All five verified claims were reproduced on finite/1-D response
   spaces with synthetic reward vectors. Nothing here says the method works on Mistral-7B,
   Alpaca-7B, UltraFeedback, or hh-rlhf. Sections 6.1 and 6.2 are untested (6.1 was not in the
   anchored claim list; 6.2 is claim 6, refused).
2. **Claim 1's "adapts to multiple alignment criteria"** was tested as "the Eq.-(1) optimum beats the
   transform-unaware optimum on Eq. (1), for m = 2,3 conflicting synthetic rewards". Per-criterion
   R_i values are stored in `results/claim1.json`, but a *single* base policy cannot beat the naive
   base on *every* criterion simultaneously and that stronger reading was neither claimed nor tested.
3. **Claim 2** verifies the discrete BoN specialisation of Prop. 4.2 and the exact prox update. The
   *practical* Algorithm 2 (clipped surrogate, ratio clipping, token-level policy gradients, group
   normalisation) is **not** tested, nor is soft BoN (Prop. 4.3), nor multi-step optimisers.
4. **Claims 3 and 4 use a numerically estimated relative-smoothness constant L** (max over 4000
   sampled policy pairs, ×1.25). It is not a proof-derived L. If the true L were larger than my
   estimate, η = 1/L would be smaller and the bound looser, so the direction of the risk is that my
   test is *harder* than the theorem, not easier; but a certified L is missing. Assumption 5.1's
   concavity half was checked only on sampled pairs (0 violations), not proved.
5. **Theorem constants are checked as inequalities, not as tightness.** I show the bounds hold
   (and break under mutation); I do not show they are the smallest valid constants, and I did not
   read or re-derive the Appendix H proofs.
6. **Claim 4's ε and δ come from injected noise** with a specific (uniform i.i.d.) shape. Real
   Monte-Carlo derivative error from M samples has a different, policy-dependent distribution; the
   O(1/M) decay of that real error is separately measured in claim 2 (slope −1.024) but was not fed
   through the Theorem-5.3 bound end-to-end.
7. **Claim 5's numerical optimum is grid-based** (K = 400 bins) and only weakly identified in TV, as
   noted; the strong evidence is the analytic first-order condition, which is necessary for optimality
   but (given β = 0 and non-strict concavity on the simplex) does not by itself establish *uniqueness*
   of the maximiser. The paper's "the optimal policy is given as" uniqueness reading is untested.
8. **Float64 only.** Claim 3's tolerance-aware pass depends on treating LHS ≤ 3.3e-16 as noise.
9. Reproduction logbooks by others were not consulted; only the paper text (once) was read, and no
   official code exists to read.
