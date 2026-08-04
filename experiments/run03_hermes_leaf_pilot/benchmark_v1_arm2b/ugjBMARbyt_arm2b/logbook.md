# Reproduction logbook — ugjBMARbyt (arm2)

**Paper:** *Linear Bandits beyond Inner Product Spaces, the case of Bandit Optimal Transport*
(L. Croissant et al.), OpenReview `ugjBMARbyt`, arXiv:2502.07397v2 (theory paper, no code release).
**Arm:** arm2 — Hermes + K-Dense skill set, **no context compaction**.
**Model:** `tencent/hy3:free` via `localhost:8319/v1`. **Hardware:** CPU-only (WSL2), numpy 2.5.1 / scipy 1.18.0.
**Global seed:** `20260803`. **Skill used:** `paper-claim-reproduction`.

**Numbering note (important).** The TASK cites Theorem 5.1 / 5.2, Corollary 5.3 / 5.4, Eq. 7 and
Eq. 11–12. The public v2 PDF renumbers these: the *content* of each anchored claim maps to
Theorem 4.1, Lemma C.3 + Thm 4.1 entropic term, Proposition 5.5, Theorem 5.3 + Corollary 5.4,
Eqs. (4)–(7) respectively. Each claim below records the exact v2 anchor it was tested against.
No original code, datasets or GPU exist for this paper, so every claim was reproduced
**from first principles** on small, exactly solvable CPU instances.

## Summary

| # | Claim (source) | Verdict |
|---|---|---|
| 1 | Fourier isometry F embeds transport plans in L²(ℝᵈ;ϱ), turning ⟨c\|π⟩ into a Hilbert inner product (v2 Sec. 3.1, Eq. (4); TASK "Sec. 4.1, Eq. 7") | **verified** |
| 2 | Trajectorial regret bound σ√(2T log(2/δ)) + 2C̄β_T(δ)√(T log det(·)) w.p. ≥ 1−δ (v2 Thm 4.1; TASK "Thm 5.1") | **verified** |
| 3 | Hölder/Lipschitz costs with ε_t = ηt^{−η}: entropic approximation is a lower-order term, regret stays sublinear (v2 Lemma C.3 + Thm 4.1; TASK "Thm 5.2") | **verified** |
| 4 | Finite basis of N coefficients ⇒ Õ(√(NT)) OFUL-style parametric rate (v2 Prop. 5.5; TASK "Cor. 5.3") | **verified** |
| 5 | ζ(n) = 1 − n^{−q} ⇒ regret interpolates between Õ(√T) and Õ(T) via exponent (q+2)/(2q+2) (v2 Thm 5.3 + Cor. 5.4) | **verified** |
| 6 | Confidence sets from regularised least squares in L²(ℝᵈ;ϱ), width driven by a log-determinant, OFUL-style ellipsoid (v2 Eqs. (5)–(7), Lemma C.2; TASK "Eqs. 11–12") | **verified** |

---

## Claim 1 — Fourier embedding (`verify_claim1.py` → `results/claim1.json`)

**Anchor:** v2 §3.1, Eq. (4): ⟨c|π⟩ = ∫ Fc(−z) Fπ(z) dϱ(z), F an isometry on L²(ℝᵈ;ϱ).

**Setup.** Discrete grid (unitary DFT is the exact discrete Fourier isometry). Checked
(a) Parseval/inner-product preservation, (b) the pairing identity for a random density
w.r.t. ϱ, (c) the pairing identity for a *genuine* coupling π ∈ Π(µ,ν) obtained by Sinkhorn
scaling (marginal error 1.4e−17).

**Numbers.** isometry relative error `1.5e−16`; norm preservation `1.6e−16`;
time-domain vs frequency-domain pairing: random density `0.2509738933825768` vs
`0.2509738933825770` (rel. gap `6.6e−16`); OT coupling `−0.05839465248784993` vs
`−0.05839465248785002` (rel. gap `1.4e−15`).

**Mutation.** Replacing F by a non-unitary, high-frequency-truncated transform breaks the
identity: relative gap `1.03e+3` (and un-normalised DFT inflates the norm by 7×).

**Verdict: verified.** The embedding claim is an exact algebraic identity and reproduces to
machine precision; the mutation confirms the test is sensitive to the isometry property.

## Claim 2 — Trajectorial regret bound (`verify_claim2.py` → `results/claim2.json`)

**Anchor:** v2 Theorem 4.1.

**Setup.** A genuine finite BOT instance: µ = ν = uniform on m = 4 points, so Π(µ,ν) is the
Birkhoff polytope whose extreme points are the 4! = 24 permutation matrices (a linear
objective is minimised at an extreme point; the Kantorovich value is the assignment optimum).
Plans are embedded as basis coefficients of dπ/dϱ (d = 16); the embedding identity
⟨f*|a_π⟩ = Σ c*_ij π_ij holds to `1.1e−16`. EntUCB/OFUL optimism with the Eq. (7) width,
λ = 1, σ = 0.2, δ = 0.05, 60 seeds, T = 400.

**Numbers.** mean realised regret `51.29`, mean certified Thm-4.1 bound `836.04`
(ratio `0.061`); the bound held in **60/60** runs (required ≥ 0.95). Uniform-in-time
coverage of f* by C_t(δ): **1.00**. Sublinearity across T ∈ {100, 400, 1600, 6400}:
per-step regret falls from `0.16` to `0.049`; log-log slopes — regret `0.757`,
certified bound `0.762` (both inflated above 1/2 on this finite range by the log factors).

**Mutation.** Replacing the σ-sub-Gaussian noise by heavy-tailed Cauchy noise (same declared σ)
destroys the assumption behind the theorem: confidence-set coverage collapses from `1.00` to
`0.00` and mean regret rises to `78.5`.

**Verdict: verified** (finite-dimensional discrete instantiation). Caveat: the entropic
approximation term of Thm 4.1 is absent here because all Birkhoff plans already lie in
Π_H(µ,ν); that term is tested separately in Claim 3. The certified bound is loose by ~16×
on this instance, so the test confirms validity, not tightness.

## Claim 3 — Entropic penalty ε_t = ηt^{−η} is lower order (`verify_claim3.py` → `results/claim3.json`)

**Anchor:** v2 Lemma C.3 (Carlier et al. 2023): Ent.(µ,ν,c,ε) − Kant.(µ,ν,c) ≤ Cε log(1/ε),
plus the entropic term of Thm 4.1.

**Setup.** Discrete OT on m = 40 points of [0,1] with the 1-Lipschitz cost |x−y|; exact
Kantorovich value by LP (`highs`, value `0.037724`), entropic value by log-domain Sinkhorn
to tolerance 1e−13, ε swept over {0.5 … 0.005}.

**Numbers.** gap = Ent(ε) − Kant is non-negative and monotone in ε
(`0.2628, 0.2024, 0.1439, 0.09164, 0.04400, 0.02312, 0.01163`); the ratio
gap/(ε log(1/ε)) stays in `[0.439, 0.758]` (spread 1.73) — i.e. the Cε log(1/ε) rate holds
with C ≈ 0.76 and is not tight from below, exactly as an upper bound should behave.
Cumulating with ε_t = ηt^{−η}, η = 0.75: `11.6, 37.8, 101.7, 245.8` at T = 10²…10⁵, matching the
predicted shape T^{1−η} log T to within a spread of `1.51`, and dominated by the √T log T
trajectorial term at all T.

**Mutation.** Constant ε (η = 0) makes the cumulated approximation error exactly **linear**:
`17.5, 174.6, 1746, 17462`, log-log slope `1.0000`.

**Verdict: verified.** Both ingredients of the claim reproduce: the ε log(1/ε) convergence rate
for a Lipschitz cost, and the sublinearity of the cumulated penalty under the decaying schedule.

## Claim 4 — Õ(√(NT)) parametric rate (`verify_claim4.py` → `results/claim4.json`)

**Anchor:** v2 Proposition 5.5.

**Setup.** Basis-truncation EntUCB (finite-dimensional OFUL with the Eq. (7) width) on
N ∈ {2,4,8,16,32}, T ∈ {250…4000}, 60 unit-norm actions, 5 repetitions per cell (125 runs).

**Numbers.** The Prop.-5.5 bound held in **125/125** runs. Stripping the logarithmic factor,
the leading term scales as exactly √N and √T: log-log slopes `0.50000` in N and `0.50000` in T,
with bound/leading-term spread `1.064` over the whole grid. Realised regret is sublinear in T
(slope `0.589` at N = 16) and increases with N (slope `1.053` at T = 2000 — steeper than the
√N of the bound on this finite range, which is consistent with an upper bound).

**Mutation.** (A) Learner truncating to 4 of 16 coefficients with tail-heavy signal: regret
inflates (`69 → 541`, slope `0.750`). (B) Breaking the linear-feedback structure (feedback
independent of the played action) yields **linear** regret: `125.5, 247.3, 487.4, 961.1, 1925.1`,
slope `0.984`.

**Verdict: verified.** The certified Õ(√(NT)) rate is reproduced exactly at the formula level
and holds empirically on every run; mutation B shows the test is not vacuous.

## Claim 5 — Interpolation between Õ(√T) and Õ(T) (`verify_claim5.py` → `results/claim5.json`)

**Anchor:** v2 Theorem 5.3 with ζ(n) = 1 − n^{−q}, and Corollary 5.4.

**Numbers.** Analytic exponent (q+2)/(2q+2) is monotone decreasing in q, equals `0.976` at
q = 0.05 (→ 1, i.e. O(T)) and `0.515` at q = 32 (→ 1/2, i.e. Õ(√T)). Numerically evaluating the
**full** Thm-5.3 expression over T ∈ {10³…10⁶} gives fitted log-log slopes
`0.982, 0.972, 0.935, 0.880, 0.807, 0.734, 0.678, 0.642, 0.611` for
q = `0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 32` — each ≥ the analytic exponent and converging to it
from above (the excess is the log factor on a finite range).
Simulation: instances with γ_i ∝ i^{−(q+1)/2} run with n_t = ⌈t^{1/(q+1)}⌉ stay below the certified
bound at every T, and the realised exponent is ordered as predicted (q = 0.1 worse than q = 2).

**Mutation.** Freezing the truncation order at n_t = 1 (never growing the basis) gives linear
regret: `37.1, 71.0, 138.8, 274.4, 545.5`, slope `0.970`.

**Verdict: verified.**

## Claim 6 — RLS confidence sets with log-det width (`verify_claim6.py` → `results/claim6.json`)

**Anchor:** v2 Eqs. (5)–(7) and Lemma C.2, with Λ = ½‖·‖², DΛ = Id.

**Numbers.** (a) The closed-form RLS estimator f̂ = (V+λI)⁻¹M*C is a stationary point of
L_t + λΛ: numerical gradient norm `3.8e−10`. (b) Sylvester identity underlying the width,
log det(Id + M*M/λ) = log det(Id + MM*/λ): gap `1.8e−15` — the width really is controlled by the
log-determinant of the design operator. (c) Uniform-in-time coverage over 400 independent runs
(d = 8, T = 300, λ = 1, σ = 0.3, δ = 0.05): **1.000** ≥ 1 − δ = 0.95, with β_T = `2.812`.

**Mutation.** Shrinking β_t by a factor 6 destroys validity: coverage `0.000`.

**Verdict: verified.** The construction is an OFUL-style ellipsoid and is (conservatively) valid;
the empirical coverage of 1.00 vs the nominal 0.95 shows the Eq. (7) width is loose, as expected
from a union-bound/self-normalised construction.

---

## Overall assessment

All six anchored claims reproduce on CPU: two are exact algebraic identities (1, 6), one is a
numerically confirmed convergence rate (3, C ≈ 0.76 for a 1-Lipschitz cost), and three are
high-probability regret bounds (2, 4, 5) that held in **every** simulated run (60/60, 125/125,
and all simulated (q,T) cells) with the certified rates matching the analytic exponents.
Every verified claim has a mutation that breaks it (non-unitary transform, Cauchy noise,
constant ε, broken feedback, frozen truncation order, shrunken β).

**Honest limitations.** (i) The paper is infinite-dimensional; all bandit experiments are
finite-dimensional/basis-truncated instantiations — they cannot falsify the infinite-dimensional
statements, only the finite instantiations of them. (ii) The regularised-optimism step (11) is
not numerically implementable in finite time (the paper says so itself, §6); we use the
extreme-point/finite-basis surrogate instead of an ε-optimal Sinkhorn plan. (iii) All bounds are
loose by one to two orders of magnitude on these instances, so this reproduces validity, not
tightness. (iv) Section/theorem numbers differ between the TASK anchors and the public v2 PDF;
the mapping is recorded per claim above.
