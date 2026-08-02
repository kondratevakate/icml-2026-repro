# logbook.md — reproduction of arXiv:2603.11919 (OpenReview MyBVUgacQ9)

**Paper:** *Last-iterate Convergence of ADMM on Multi-affine Quadratic Equality Constrained
Problem* — Chao, Ciebielski, Etesami, Khadiv (TUM), math.OC, 12 Mar 2026.
**Reproducer:** autonomous agent, CPU only (`.venv`: numpy / scipy / sympy).
**Official code:** none. The PDF contains no repository link (`grep -i github|code available` → 0 hits),
so every experiment here was implemented from the paper's own equations
(Algorithm 1, eqs. 1–3, eq. 4, eq. 6, Section 5 toy problem).

**Time budget:** first tool call 2026-08-02 00:30 local; logbook completed 01:35 local.
**Elapsed ≈ 1 h 05 min** (hard stop 8 h, soft target 4 h, per-claim cap 2 h — all respected;
the longest single claim, claim 2, took 2191 s of compute).
**LLM turn count:** 22 turns for the whole paper (budget 60).

Shared implementation: `admm_lib.py` — exact-block ADMM (Algorithm 1) for problem (1) with
closed-form block updates (1-D/small strongly convex quadratics; box-constrained blocks solved
exactly by active-set enumeration), so the "sub-problems solved exactly" hypothesis of
Theorems 3.1–3.3 holds by construction.

**Rate diagnostic used throughout.** For each run we take the iterate-change sequence
`res_k = ||(x,z)^{k+1} − (x,z)^k||` (and, for claim 3, the exact Theorem-3.3 gap) and fit
`log res_k` linearly over the tail above the 1e-13/1e-14 numerical floor. The fitted slope gives
the **per-iteration contraction factor** `c^{-1}`: a value bounded away from 1 ⇔ linear
(geometric) convergence; a value → 1 ⇔ sublinear. Every claim is evaluated over a **grid of
initialisations / seeds (12–450 runs per claim), never a single seed.**

---

## Claim 1 — Theorem 3.1 (Section 3): sublinear convergence to a stationary point

**Verdict: `verified`.**

*Source:* Theorem 3.1, Section 3 (with Assumptions 2.3, 2.6, Definition 2.7 / eq. 3, Algorithm 1).
*Script:* `verify_claim1.py` → `results/claim1.json` (command recorded in the JSON).

Numbers (450 runs = toy problem q ∈ {0.5, 2, 5} × 3 ρ values ≥ the Theorem-3.1 threshold ×
16 x-initialisations × 3 z-initialisations, plus Example 2.2 with 9 initialisations × 2 ρ):

| quantity | value |
|---|---|
| runs converged (`res < 1e-8`) | **450 / 450** |
| runs feasible (`‖A(x)+Qz‖ < 1e-8`) | **450 / 450** |
| max final residual | **3.846e-16** |
| max final constraint violation | **2.220e-16** |
| o(1/k) witness `max_k k·(L^k − L*)` on the tail | **0.000e+00** (450/450 runs below 1e-6) |
| max blockwise (Nash-like) optimality error of the limit point | **5.192e-10** |

The last row is the substantive part of Theorem 3.1: for every block *i* we solved the paper's
characterisation `x*_i ∈ argmin f(x_i, x*_{−i}) s.t. A(x_i, x*_{−i}) + Q z* = 0` exactly and
compared with the ADMM limit — agreement to 5e-10. The observed rate is in fact geometric, which
is consistent with the theorem's "**at least** sublinear".

**Mutation A (mechanism: Assumption 2.6, Q full row rank).** Example 2.8 (`min x²+y² s.t. xy=1`,
Q = 0 ⇒ not full row rank), 12 runs (4 initial points × 3 ρ): iterates collapse to the origin
(`max‖x_final‖ = 0.0`), the dual diverges (`w = −4000` at every setting, monotone in ρ·k), and the
limit is **infeasible** (`violation = 1.000` in all runs). Removing Assumption 2.6 therefore
destroys exactly what Theorem 3.1 delivers — the theorem's hypothesis is load-bearing.

**Mutation B (the ρ threshold).** With `ρ_threshold = 500` for a problem with L_φ = μ_φ = 5,
Q = [0.2]: ρ = 1e-4·threshold … 0.1·threshold all still converge (worst residual < 1e-12) while
ρ ≥ threshold converges more slowly (worst residual 1.2e-4 after 1500 iterations). This does **not**
contradict Theorem 3.1 (its ρ bound is sufficient, not necessary) but it is honest evidence that
the stated bound is conservative and not the mechanism driving convergence in practice.

---

## Claim 2 — Theorem 3.2 + Equation 4 (Section 3): linear rate for small ‖C‖

**Verdict: `verified` (qualitative content of eq. 4; the constants m1,m2,m3 are not numerically
specified in the paper, so the exact threshold cannot be tested).**

*Source:* Theorem 3.2 and Equation (4), Section 3. *Script:* `verify_claim2.py` → `results/claim2.json`.

Setup: toy problem with the nonlinear part scaled, `s·(x1x2 − x3x4) + q z + 1 = 0`, q = 2 fixed
(so Q is fixed, as eq. 4 requires), ‖C‖ = s swept; 48 initialisations × 3 ρ per cell; the
limit point's reduced Hessian was checked (second-order differentiability / local-minimum part).

| ‖C‖ | best-ρ worst-case contraction factor | regime |
|---|---|---|
| 0 (linear constraints) | **0.0588** | Thm 3.2 region (reference) |
| 0.01 / 0.05 / 0.2 / 0.5 | 0.0589 / 0.0589 / 0.0588 / 0.0588 | Thm 3.2 region |
| 1 | **0.1111** | Thm 3.2 region |
| 2 | 0.2507 | mutation arm |
| 5 | 0.8058 | mutation arm |
| 10 | 0.8585 | mutation arm |
| 25 | **0.9981** | mutation arm |
| 50 | **1.0004** (no contraction; final violation 7.7e-05) | mutation arm |

So over the whole small-‖C‖ range the rate is linear and essentially identical to the ‖C‖ = 0
(purely linear-constraint) case — exactly the claim "as long as ‖C‖ is small enough, linear
convergence is still preserved", including the ‖C‖ → 0 consistency with Lin et al. (2015b).

**Mutation (mechanism: eq. 4).** Inflating ‖C‖ past the small regime degrades the factor
monotonically and destroys linear convergence entirely at ‖C‖ = 25–50 (factor ≥ 0.998, and at
‖C‖ = 50 the iterates no longer even reach feasibility, violation 7.7e-05). The rate therefore
tracks the ‖C‖-vs-Q balance that eq. (4) posits, rather than being a property of ADMM per se.

---

## Claim 3 — Theorem 3.3 (Section 3): linear rate with polyhedral indicators, no second-order differentiability

**Verdict: `verified`.**

*Source:* Theorem 3.3, Section 3. *Script:* `verify_claim3.py` → `results/claim3.json`.

Setup: toy problem + **box** constraints `x_i ∈ [0.3, 1.5]` (boxes are polyhedra) chosen so that
the limit sits on an active face; 12 seeds per cell; the Theorem-3.3 quantity
`L(x^k,z^k,w^k) − min_{(x,z)∈B((x^k,z^k);r)} L(x,z,w^k)` with r = 0.05 was evaluated exactly by
SLSQP over box ∩ ball.

| cell (q, ρ) | worst residual factor | worst Thm-3.3 gap factor | active constraints at limit | local-min probe failures |
|---|---|---|---|---|
| q=2, ρ=1 | **0.2000** | **0.0522** | 4 / 4 | 0 / 12 |
| q=2, ρ=4 | 0.0668 | **0.0881** | 4 / 4 | 0 / 12 |
| q=5, ρ=1 | 0.0385 | 0.0056 | 4 / 4 | 0 / 12 |
| q=5, ρ=4 | 0.0273 | 0.0818 | 4 / 4 | 0 / 12 |

All four box constraints are active at every limit point, i.e. the Lagrangian is **not**
second-order differentiable there and Theorem 3.2 does not apply — yet the Theorem-3.3 gap decays
geometrically with per-iteration factor ≤ 0.0881, and 4×12×400 = 19 200 feasible perturbation
probes found **no** better point than the limit (local minimality, second part of Thm 3.3).

**Mutation A (eq. 4 broken while keeping polyhedrality).** ‖C‖×10 → residual factor 0.9858 and
gap factor 0.9977; ‖C‖×50 → residual factor 1.0006 (no contraction at all). The polyhedral
structure alone does not buy the rate; eq. (4) is still needed, as the theorem states.

**Mutation B (polyhedrality broken, eq. 4 kept).** Replacing the box by a Euclidean ball
(exact projection) still yields a geometric factor ≈ 0.0625 across 12 seeds. This is reported
as-is: Theorem 3.3 is a sufficient condition, and our evidence does not show polyhedrality to be
necessary — it only shows the theorem's conclusion holds inside its stated hypothesis.

---

## Claim 4 — Corollary 4.2 (Section 4): locomotion, linear rate for Δt ≤ t0 because nonlinearity is O(Δt³)

**Verdict: `verified`** for the O(Δt³) mechanism and for linear convergence of eq. 6 at small Δt;
the *specific* threshold t0 (the paper's own "Δt = 0.005 s, and the bound is conservative") is not
pinned down — see Evidence boundary.

*Source:* Corollary 4.2 and eq. 6, Section 4 (with Fig. 5's parameters m = 2 kg,
f(f)=½Σ‖f_i‖², φ(z)=5Σ‖k'_i‖²). *Script:* `verify_claim4.py` → `results/claim4.json`.

**(A) Symbolic (sympy).** Independently deriving eq. 6 from the eq.-5 recursion for a planar
instance (T = 4, N = 2 contacts) and splitting the expansion by degree in the forces:

* rows 1–2: 0 quadratic terms; rows 3–4: 8 and 16 quadratic terms, and **every** quadratic
  coefficient has Δt-exponent exactly **3** (`all_quadratic_dt_pow3 = true` for all rows), while
  linear terms carry Δt-exponents {1, 2, 3}.
* the C_i / d_i matrices built by our numeric eq.-6 model match the sympy expansion to
  **2.78e-17** (max abs error over 5 random force vectors).
* fitting ‖C‖ ∝ Δt^p over Δt ∈ {0.002 … 0.05} gives **p = 3.0000**.

This is the paper's stated reason ("the nonlinear term is proportional to (Δt)³"), reproduced exactly.

**(B) Simulation.** Algorithm 1 on eq. 6 (T = 6, N = 2, m = 2 kg, polyhedral box force
constraints, ρ = 4, 4 random initialisations each):

| Δt | ‖C‖ | worst contraction factor | worst final violation |
|---|---|---|---|
| 0.002 | 4.38e-08 | **0.7143** | 7.1e-18 |
| 0.005 | 6.85e-07 | 0.7143 | 8.0e-18 |
| 0.010 | 5.48e-06 | 0.7143 | 1.1e-17 |
| 0.020 | 4.38e-05 | 0.7142 | 1.1e-17 |
| 0.050 | 6.85e-04 | 0.7138 | 2.2e-17 |
| 0.100 | 5.48e-03 | 0.7106 | 4.5e-17 |

Linear convergence at every Δt tested, including well above the paper's suggested 0.005 s —
consistent with the paper's own remark that the corollary's bound is conservative.

**Mutation (mechanism: the Δt³ decay).** Rescaling *only* the quadratic block by Δt⁻³ so that
‖C‖ = 5.477 independently of Δt: the contraction factor jumps to **0.9997 at every Δt** and
feasibility degrades to 4e-05…2e-04. So it is genuinely the O(Δt³) decay of the nonlinearity —
not the smallness of Δt in the linear terms — that produces linear convergence, exactly as
Corollary 4.2 argues.

---

## Claim 5 — Section 5, Figures 2 and 4 (baselines + "q ≥ 10 gives the sublinear→linear transition")

**Verdict: `inconclusive`** (split, and *not* supported as stated for the q ≥ 10 half).

*Source:* Section 5, Figure 2 (toy problem, "Condition in (4) suggests q ≥ 10 to ensure linear
convergence") and Figure 4 (comparison with PADMM, IPDS-ADMM, IADMM).
*Script:* `verify_claim5.py` → `results/claim5.json`.

**(i) q ≥ 10 transition.** The toy problem is fully specified in Section 5 but μ_x, μ_z and ρ are
not reported, so we swept q over 10 values × 6 hyperparameter settings × 6 seeds and took, per q,
the **best** (most favourable to the claim) worst-case contraction factor:

| q | 0.5 | 1 | 2 | 5 | 8 | 10 | 12 | 15 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| best worst-case factor | 0.2296 | 0.1667 | 0.0475 | 0.0378 | 0.0588 | 0.0317 | 0.0250 | 0.0245 | 0.0287 | 0.3364 |

**All 5 tested values of q < 10 are already linear** (factors 0.03–0.23; median below q=10 is
0.0588 vs 0.0287 at/above q=10 — the same order of magnitude), and q = 50 is *worse* than q = 10.
We found **no sublinear regime below q = 10 and hence no transition at q = 10** under any of the
six hyperparameter settings. Under a *fixed* ρ the ordering is in fact reversed (larger q slower).
Because the paper does not report μ_x, μ_z, ρ or the initialisation of Figure 2, we cannot exclude
a setting in which the figure's transition appears, so the honest verdict is `inconclusive` rather
than `falsified`: the specific numeric threshold "q ≥ 10" is **not reproducible from the
information published**.

**(ii) Baseline comparison (Figure 4): `inconclusive`, deliberately no numbers.** No code release,
and the paper reports neither the problem instances/dimensions of the three panels, nor the
baselines' hyperparameters (proximal / dual step sizes for PADMM, IPDS-ADMM, IADMM), nor
initialisations or seeds. Any re-implementation would benchmark our own baseline choices, not the
authors', so no comparison figure was produced.

---

## Claim 6 — Section 5, Figures 5 and 6: 2D locomotion simulation + real robot experiments

**Verdict: `inconclusive`** (hardware unavailable; no toy substitute produced).

*Source:* Figures 5 and 6 (and Figure 3). *Script:* `verify_claim6.py` → `results/claim6.json`.

* Humanoid vertical jump and quadruped bounding (Fig. 6) require physical robots; no code, robot
  model, logs or rosbags were released, and the tracking layer is an external DDP implementation
  (Crocoddyl) with unreported cost weights. Figure 3's dynamic-violation curves come from that
  same unreleased pipeline. Not reproducible here, and no simulated stand-in was presented as if
  it were the hardware result.
* The 2D-locomotion-simulation half is *partially* covered by claim 4 (eq. 6 implemented, linear
  convergence observed across Δt, forces box-constrained), but the specific curves of Figure 5
  cannot be matched because T, the target CoM area, the contact schedule and locations r_i^j, the
  initial state and ρ are not reported.

---

## Final summary table

| Claim | Source | Verdict | One-line evidence |
|---|---|---|---|
| 1 | Thm 3.1, §3 | **verified** | 450/450 runs converge to feasible, blockwise-optimal limits (max blockwise error 5.19e-10, k·gap = 0); mutating Assumption 2.6 (Example 2.8) gives x→0, w = −4000, violation 1.000 |
| 2 | Thm 3.2 + eq. 4, §3 | **verified** | contraction factor ≤ 0.111 for ‖C‖ ∈ [0, 1] (equal to the ‖C‖=0 linear-constraint reference 0.0588); mutation to ‖C‖ = 25/50 gives 0.9981 / 1.0004 (linear rate lost) |
| 3 | Thm 3.3, §3 | **verified** | with all 4 polyhedral constraints active at the limit (Lagrangian nonsmooth there), the Thm-3.3 gap contracts at ≤ 0.0881 per iteration and 19 200 probes confirm local minimality; ‖C‖×10 mutation → 0.9977 |
| 4 | Cor 4.2 + eq. 6, §4 | **verified** | sympy expansion of eq. 6: every quadratic coefficient is exactly Δt³ (fitted p = 3.0000, model-vs-symbolic 2.8e-17) and ADMM is linear (factor 0.714) for Δt ∈ [0.002, 0.1]; forcing Δt-independent nonlinearity → 0.9997 |
| 5 | Figs. 2 & 4, §5 | **inconclusive** | no sublinear regime found below q = 10 (all 5 tested q<10 linear, best factors 0.03–0.23), so the "q ≥ 10" threshold is not reproducible from published information; Fig. 4 baselines have no code/hyperparameters → no numbers produced |
| 6 | Figs. 5 & 6, §5 | **inconclusive** | humanoid-jump and quadruped-bounding results need physical hardware and an unreleased DDP pipeline; only the eq.-6 mechanism (claim 4) could be reproduced |

Reproducibility gate: `.venv/bin/python check_reproducibility.py` → **38 assertions passed,
0 failed** (every number quoted above is re-asserted against `results/claim{1..6}.json`).

---

## Evidence boundary — what this evidence does NOT cover

1. **No proofs were checked.** All theory claims were tested *empirically* on instances that
   satisfy the hypotheses. A verified verdict here means "the theorem's conclusion holds and its
   hypotheses are load-bearing on the instances tested", not "the proof in the appendix is correct".
   Appendices A–E (formal subanalyticity/KL arguments, the approximated-ADMM of Appendix D.1, the
   polyhedral extension of Appendix E) were not examined at all.
2. **Constants are untested.** Eq. (4) contains unspecified constants m1, m2, m3 and Theorem 3.2/3.3
   assert only the existence of c1, c2 > 1. We could test the *qualitative* small-‖C‖ statement and
   measure contraction factors, but not the *threshold* value of ‖C‖, nor c1, c2, nor r in Thm 3.3
   (we fixed r = 0.05).
3. **Rate classification is numerical.** "Linear" here means a per-iteration contraction factor
   fitted below ~0.999 over the range above the 1e-13/1e-14 floating-point floor; "not linear"
   means factor ≥ 0.999 within the iteration budget (250–1500 iterations). We cannot distinguish
   asymptotically-linear-with-factor-0.9999 from genuinely sublinear.
4. **Instance coverage is narrow.** Claims 1–3 and 5 use the paper's own 4-block scalar toy problem
   (plus Example 2.2); claim 4 uses a planar locomotion instance of our own design (T = 4–6, N = 2
   contacts, box force constraints as the polyhedral cone approximation). No large-scale, no
   ill-conditioned, and no high-dimensional instances; no matrix-factorisation or neural-network
   instances despite the paper citing them as applications.
5. **Sub-problems are solved exactly.** We never tested the inexact/approximated-ADMM variant of
   Appendix D.1, which is what a practical implementation would use.
6. **Exhaustiveness is over initialisations, not over problem space.** 450 (claim 1), 48×11
   (claim 2), 12 per cell (claim 3), 4 per Δt (claim 4), 6 per cell (claim 5) initialisations —
   sufficient to exclude single-seed artefacts, insufficient to make worst-case statements.
7. **Figures 2, 3, 4, 5 are not numerically reproduced.** The paper does not publish the
   hyperparameters (μ_x, μ_z, ρ, T, initialisations, baseline step sizes) needed to redraw them;
   our sweeps replace them and therefore answer a weaker question.
8. **Nothing about hardware.** No claim about real-robot feasibility, tracking accuracy, or the
   real-time behaviour of the method on a humanoid/quadruped is supported by anything here.
9. **Assumption 2.3's subanalyticity/α-PL machinery** was assumed satisfied by our smooth quadratic
   instances; it was not verified for the general problem class.
