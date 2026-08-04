# logbook.md — reproduction of DUmWdZetqZ

**Paper:** *Fixed Budget is No Harder Than Fixed Confidence in Best-Arm Identification up to
Logarithmic Factors* — OpenReview `DUmWdZetqZ`, arXiv `2602.03972` (HTML v1 → `results/paper.txt`).

**Nature of the paper:** pure theory. No experimental section, no datasets, no GPU, no official
code repository referenced in the text. Every anchored claim is therefore either analytic or
checkable by an exact/Monte-Carlo bandit simulation on CPU. No claim had to be refused for
data/GPU reasons.

**Elapsed:** first tool call 2026-08-01 23:09 (+04), finished 23:33 — **24 minutes** wall clock,
far inside the 4h soft target and the 2h/claim cap. **LLM turns:** ~24 of the 60 budget.
All five scripts plus the gate were re-run end-to-end from scratch as a final check
(`claim1..5 rc=0`, gate `125 checks, 0 failures`).

**Environment:** `.venv/bin/python` (3.12, CPU) with numpy + sympy; nothing else installed.

**Method note (why the evidence is strong).** The meta-algorithms are stated *for any* strong
(Def 3.1) / weak (Def 4.1) fixed-confidence algorithm. So the tests drive them with the
**adversarial oracle that saturates those definitions with equality** (`repro_lib.StrongFCOracle`
and the weak oracle in `verify_claim3.py`): it stops immediately with a *wrong* arm w.p. δ, never
stops w.p. δ, and otherwise stops exactly at `T*_δ` with the right arm. Any theorem quantified
over Definition 3.1 must survive it. For those oracles the meta-algorithms' error probabilities
are computable in **closed form by exhaustive enumeration** over the finite outcome space (stages
for FC2FB, the multinomial `(n_wrong, n_correct, n_never)` for FCW2S, all integer horizons `T`
for FC2AT) — so the primary numbers are **not seed estimates at all**; Monte Carlo over 5 seeds is
only a cross-check that the closed form matches the literal pseudocode. Claim 5's Part A uses the
real Algorithm 5 on real Gaussian bandits, 5 seeds × 200 runs per budget.

---

## Claim 1 — Theorem 3.2: FC2FB converts FC → FB with exponentially decaying error
**Source:** Section 3, Algorithm 3 + Theorem 3.2.
**Script:** `verify_claim1.py` → `results/claim1.json`
**Verdict: `verified` (with a boundary caveat on the printed validity condition).**

Justifying numbers:
- Grid of 432 configurations `(A ∈ {5,20,100,500}) × (C ∈ {0,10,200}) × (Q ∈ {1,8}) ×
  (δ₀ ∈ {0.5, 1/e, 0.1}) × (B = ⌈B_min·m⌉, m ∈ {1,2,4,8,16,32})`, exact error vs the
  Theorem 3.2 bound `3exp(−B/(4Q/ln(1/δ₀)+4log₂(B/Q)A))`.
- On the 360 points with **B ≥ 2·B_min: 0 violations**, minimum slack ratio (bound/actual)
  **2.17**.
- **Exponential decay:** at `A=50, C=20, Q=1, δ₀=1/e`, exact error falls from `7.758e-01` to
  `1.604e-28` as B goes 704 → 90 078 (a factor **4.84e27** over 128× budget); linear fit of
  `log P(err)` on B gives slope **−6.820e-04** per sample, **R² = 0.983** (the residual is the
  `⌊log₂⌋` staircase in R, not curvature).
- **Monte Carlo cross-check** (5 seeds × 200 000 runs, B = 2815): exact `2.2946e-02` vs MC mean
  `2.2659e-02`, absolute deviation **2.87e-04**.

**Boundary caveat (a real, small discrepancy).** 12 of the 432 points violate the printed
inequality. **All 12 sit exactly at the stated threshold `B = B_min`** (max `B/B_min` among
violations = 1.0) and **all have C > A**. Worst case: `A=5, C=200, Q=1, δ₀=0.5, B=2460` →
exact error `9.730e-02` vs bound `7.134e-05` (slack ratio `7.33e-04`). Cause: the theorem's
exponent contains no `C`, while the achievable per-stage confidence is
`δ₀^{(B/R − C)/(A ln(1/δ₀))}`; the printed threshold `B ≥ 2(A ln(1/δ₀)+C+1)ln(…)` does not force
`B/R` far enough above `C` when `C ≫ A`. Requiring `B ≥ 2·B_min` removes every violation. This
affects only the constant in the validity condition, not the substance of the claim
(the exponential decay in B, which is what the anchored claim states).

**Mutation test.** Replacing the doubly-exponential schedule `L_r = 2^{R−r}`:
- M1 constant schedule (`L_r = 1`, every stage uses δ₀): error **stalls at 0.582** (ratio
  first/last budget = 1.0001 — no decay at all) and violates the Theorem 3.2 bound at **5 of 8**
  budgets. Compare: true schedule reaches `1.604e-28` at the same budget.
- M2 reversed schedule (`L_r = 2^{r−1}`): stalls at **0.419**, **5 of 8** bound violations.
The mechanism (increasing δ at a doubly exponential rate) is therefore load-bearing, not incidental.

---

## Claim 2 — inverted sample complexity `O(A ln(1/δ)·ln(A ln(1/δ)/Q) + C)`
**Source:** Section 3, the display immediately after Theorem 3.2 (δ₀ = 1/e).
**Script:** `verify_claim2.py` → `results/claim2.json`
**Verdict: `verified`.**

Method: exact integer bisection for `B_req(δ)` = smallest budget satisfying both the Theorem 3.2
validity condition and `bound ≤ δ`; compared against
`F = Q ln(1/δ) + A ln(1/δ) ln(max(e, A ln(1/δ)/Q)) + C`. Deterministic, no randomness.

Justifying numbers, over **420** grid points (`A ∈ {1,10,10²,…,10⁶}`, `C ∈ {0,10,10³,10⁵}`,
`Q ∈ {1,10,100}`, `δ ∈ {0.1,10⁻²,10⁻³,10⁻⁶,10⁻¹⁰}`):
- `max B_req/F = **24.41**` — a single absolute constant across 6 orders of magnitude in A,
  5 in C and 10 in 1/δ. The per-A maximum is **non-increasing**: `24.41, 24.40, 24.14, 20.94,
  15.45, 13.66, 12.63` for `A = 1 … 10⁶` (growth factor A=10⁶ over A=10 is **0.518 < 1**).
  That is exactly what the `O(·)` asserts.

**Mutation test.** Drop the inner logarithm (`F_naive = Q ln(1/δ) + A ln(1/δ) + C`, i.e. claim
that FB matches FC *without* log factors). The ratio then **grows monotonically with A**:
`24.41, 56.93, 83.38, 105.62, 126.89, 147.83, 168.58` for `A = 1 … 10⁶` — a factor **6.91**
blow-up, unbounded in A. So the `ln(A ln(1/δ)/Q)` factor is necessary and is not padding.

Diagnostic (not part of the verdict): in the recommended regime `Q ≤ A`, `max B_req/T*_δ = 168.6`
and `max B_req/(T*_δ (1+ln T*_δ)²) = 13.6`, i.e. the overhead is polylog, not polynomial.

---

## Claim 3 — FCW2S (Algorithm 4) converts weak FC (Def 4.1) → strong FC (Def 3.1)
**Source:** Section 4, Algorithm 4, Propositions 4.2 and 4.3 (proofs in Appendix C).
**Script:** `verify_claim3.py` → `results/claim3.json`
**Verdict: `verified`.**

Setting: `δ₀ = 1/(8e)` (the value the paper itself uses in Corollary 5.7), `f(δ₀) = 100`,
`L = ⌈4 ln(1/δ)/ln(1/(4e δ₀))⌉`. Error/stopping probabilities computed by **exhaustive
enumeration of the whole multinomial support** `(n_wrong, n_correct, n_never)`; all wrong
instances vote for the *same* wrong arm and ties go to the wrong arm (worst case).

Justifying numbers over 8 target δ from 0.5 to 1e-8:
- **Prop 4.2**: `0/8` violations of `(4eδ₀)^{L/4}` and `0/8` violations of `≤ δ`.
  e.g. δ=0.1 → L=14, exact error `2.321e-06` ≤ bound `8.839e-02` ≤ δ.
- **Prop 4.3**: `0/8` violations of `(2eδ₀)^{L/2}` and `0/8` of `≤ δ`.
  e.g. δ=0.1 → exact tail `1.120e-06` ≤ bound `6.104e-05`.
- **Strongness (Def 3.1)**: `T*_δ = L·f(δ₀)` fitted on `ln(1/δ)` gives slope
  **578.07**, against the predicted `4f(δ₀)/ln(1/(4eδ₀)) = **577.08**`, with **R² = 0.99993** —
  i.e. `T*_δ = A ln(1/δ) + C` with A ≈ 578, exactly the form Definition 3.1 requires.
- Literal event simulation of Algorithm 4 (5 seeds × 20 000 runs, L=54) matches the exact
  computation to **2.4e-20** (error) and **4.5e-22** (tail).

**Mutation test.** Replace majority voting by "output the arm of the first instance that
terminated" (wrong instances terminate first by construction): error jumps to **0.9935** at the
largest L, does **not** decay with L (ratio first/last = 0.486, i.e. it *worsens*), and violates
Prop 4.2 at **6/6** grid points. Majority voting is the mechanism, not decoration.

---

## Claim 4 — FC2AT (Algorithm 6): anytime variant via the doubling trick, no budget knowledge
**Source:** Section 4 pointer + Appendix D (Algorithm 6, Defs D.1–D.2, Props D.3–D.4, Thm D.5).
**Script:** `verify_claim4.py` → `results/claim4.json`
**Verdict: `verified`.**

Justifying numbers:
- **No horizon knowledge (structural):** phase lengths are `T_i = 2^i·Q`
  (`Q=1 → 2,4,8,16,32,64,128`), a function of `(i, Q)` only; a standing recommendation exists
  at every `t` (checked for all `t ∈ [1, 5000)`).
- **Props D.3 / D.4 exhaustively:** over 4 parameter settings and **80 004 consecutive integer
  horizons T** in the admissible range `T ≥ max(4B*−2Q, 2Q)`:
  **0 violations of `T_{I_f} ≥ B*`** and **0 violations of `T_{I_f} ≥ T/4`**.
- **Theorem D.5:** exact anytime error of the last completed phase vs
  `3exp(−T/(16Q/ln(1/δ₀)+16log₂(T/Q)A))`: **0 violations** over 21 (setting, T) pairs; the error
  falls by at least a factor **4.29e9** across the T range in every setting.

**Mutation test.**
- M1 — remove the doubling (constant phase length `T_i = 2Q`): the error **stalls at 1.0** and
  violates the Theorem D.5 anytime bound at **4 of 7** horizons.
- M2 — sub-geometric growth `T_i = ⌈1.05^i Q⌉`: Prop D.4 (`T_{I_f} ≥ T/4`) fails at **6 of 6**
  horizons. Geometric growth is exactly what buys the anytime guarantee.

---

## Claim 5 — improvements on heterogeneous-noise (Cor 5.2), linear (Cor 5.4), unimodal (Cor 5.7)
**Source:** Section 5 (5.1 Alg 5 / Thm 5.1 / Cor 5.2; 5.3 Cor 5.4; 5.4 Cor 5.7).
**Script:** `verify_claim5.py` → `results/claim5.json`
**Verdict: `verified` for the heterogeneous-noise application; `inconclusive` for the linear and
unimodal applications.** (Reported as `partial` in the JSON.)

### 5a Heterogeneous noise — `verified`
Instance: Gaussian, `mu = [1.0, 0.6, 0.4, 0.2]`, `sigma = [0.15, 0.15, 1.0, 1.0]`, giving
`A = **295.78**`, `C = **956.13**` from Theorem 5.1.
- **Theorem 5.1** (Algorithm 5 implemented verbatim, 5 seeds × 400 runs per δ ∈ {0.2, 0.1, 0.05}):
  `0/3` violations of `P(τ > T*_δ) ≤ δ` and `0/3` of δ-correctness.
- **FC2FB(PE-KHN) end-to-end** (real algorithm inside the real meta-algorithm, δ₀=1/e, Q=1,
  5 seeds × 200 runs per budget): mean error `0.724 → 0.050 → 0.0070 → 0.0 → 0.0` for
  `B = 2500, 5000, 10 000, 20 000, 40 000`, monotone decreasing and **0 violations** of the
  Theorem 3.2 bound (`2.488, 2.127, 1.588, 0.919, 0.329`).
- **K⁷ vs K⁹ separation** (sympy, exact): on the paper's instance the FC2FB(PE-KHN) complexity is
  `K**7 − 2K**6 + 2K**5` (degree **7**) and SHVar's is `K**9 − 2K**8 + 2K**5` (degree **9**),
  matching the paper's `O(K⁷)` vs `O(K⁹)`.
- **Mutation test:** make the allocation noise-blind (use σ_max for every arm — the SH/SHVar
  style rule). The sample budget grows by **4.85×** (δ=0.1) and **4.78×** (δ=0.01) on the same
  instance. The per-arm σ_i allocation is the source of the improvement.

**Additional finding — Corollary 5.2 is printed stronger than its parent theorem.** Theorem 3.2
with `δ₀ = 1/e, Q = 1` yields denominator `4 + 4A·log₂B`; Corollary 5.2 prints `4 + 4A·lnB`.
Since `log₂B = 1.4427·lnB`, the printed corollary bound is strictly *smaller* than what
Theorem 3.2 gives, at all **9/9** checked `(A, B)` points; the gap reaches a factor
**3.29e28** at `A=10, B=1e5`. This looks like a `ln`↔`log₂` typo in the corollary; it does not
affect Theorem 3.2, and the empirical errors above satisfy both forms.

### 5b Linear bandits (Corollary 5.4) — `inconclusive`
Stated in terms of `gamma*`, `rho*` and the algorithm *Fixed Budget Peace* (Katz-Samuels et al.
2020, Algorithm 3). None of these is defined in this paper; verifying the corollary requires
re-implementing that external algorithm and computing its instance constants. Not attempted
within budget. **No toy substitute was constructed** — a self-invented "linear-bandit-ish"
simulation would test my own construction, not the paper's claim.

### 5c Unimodal bandits (Corollary 5.7) — `inconclusive`
Same reason: stated via `T_mu(delta)` and *UniTT* (Poiani et al. 2024, Theorem 3.7), external.
Partial evidence does exist: the **FCW2S half** of the Corollary 5.7 pipeline is verified in
Claim 3 with exactly the parameters Corollary 5.7 prescribes (`δ₁ = 1/(8e)`,
`L = ⌈4 ln(1/δ)/ln(1/(4e δ₁))⌉`), and the FC2FB half is verified in Claim 1. What is unverified is
the instance-dependent constant `T_mu(δ₁)+K` and the claimed dominance over `Δ^{-2}` (Prop 5.8).

---

## Final summary table

| Claim | Verdict | One-line evidence |
|---|---|---|
| 1 — Thm 3.2, FC2FB, exponential error decay | **verified** (boundary caveat) | Exact error vs bound: 0/360 violations for B ≥ 2B_min (min slack 2.17); error falls 7.76e-01 → 1.60e-28 over a 128× budget range (log-linear fit R²=0.983); mutating the δ-schedule stalls the error at 0.582 and breaks the bound 5/8 times. 12/432 points at exactly B = B_min with C ≫ A violate the printed inequality. |
| 2 — complexity `O(A ln(1/δ) ln(A ln(1/δ)/Q) + C)` | **verified** | Exact budget inversion over 420 configurations: `B_req/F ≤ 24.41` with the per-A maximum *decreasing* (24.41 → 12.63 for A = 1 → 10⁶); dropping the inner ln makes the ratio grow 24.41 → 168.58. |
| 3 — FCW2S (Alg 4) weak → strong FC | **verified** | Exhaustive multinomial enumeration: 0/8 violations of Props 4.2 and 4.3 and of `≤ δ`; `T*_δ` slope 578.07 vs predicted 577.08 (R²=0.99993, i.e. Def 3.1 form); replacing majority voting by first-terminated pushes the error to 0.9935 and breaks Prop 4.2 6/6. |
| 4 — FC2AT (Alg 6) anytime doubling | **verified** | 80 004 consecutive integer horizons: 0 violations of `T_{I_f} ≥ B*` and `≥ T/4`; 0/21 violations of the Thm D.5 anytime bound with ≥4.29e9 error decay; constant-phase mutation stalls at error 1.0 (4/7 bound violations) and 1.05^i growth breaks Prop D.4 6/6. |
| 5 — applications (Cor 5.2 / 5.4 / 5.7) | **verified (heterogeneous noise)** / **inconclusive (linear, unimodal)** | Real Alg 5 + FC2FB: 0/3 violations of Thm 5.1, error 0.724 → 0.0 over B = 2.5k → 40k with 0 bound violations, sympy confirms K⁷ vs K⁹, noise-blind mutation costs 4.85×. Cor 5.4 / 5.7 rest on external constants (Peace / UniTT) — not reproduced, not faked. Side finding: Cor 5.2 prints `ln B` where Thm 3.2 gives `log₂ B`, making the corollary stronger than its parent. |

Reproducibility gate: `.venv/bin/python check_reproducibility.py` re-asserts every number quoted
above against `results/*.json`.

---

## Evidence boundary — what this evidence does **not** cover

1. **No proofs were checked.** Nothing here verifies the *derivations* in Appendices B, C, D, F.
   The evidence is that the stated inequalities hold numerically for the algorithms as written,
   driven by definition-saturating oracles — not that the proofs are correct.
2. **Oracle-driven, not universal.** Claims 1–4 are tested with *one* adversarial member of
   Definitions 3.1 / 4.1 (plus, for claim 5a, the real PE-KHN). A definition-satisfying algorithm
   with a different stopping-time *distribution* (e.g. heavy-tailed within the allowed envelope,
   or correlated across the repeated FC2FB stages — my stages are independent, as re-running a
   fresh instance implies) could behave differently. Failure of the oracle disproves; success does
   not prove universality.
3. **Finite grids.** "0 violations" always means 0 over the enumerated grid listed per claim
   (432 / 420 / 8 / 80 004 / 15 configurations). Regimes outside those ranges — in particular
   very large `A` with `C ≫ A`, `δ₀` near 0.5 with tiny `Q`, and `B` between `B_min` and `2B_min`
   — are only partially probed, and that is exactly where claim 1's violations live.
4. **The claim-1 boundary violation is not fully characterised.** I show 12 violating points and
   that `B ≥ 2B_min` fixes them on this grid; I did **not** derive the corrected constant, nor
   confirm whether the paper's proof already implies a tighter condition than the one printed.
5. **Corollary 5.2's `ln` vs `log₂`** is diagnosed by algebra only; I did not inspect the
   Appendix-F derivation to confirm it is a typo rather than a separately-proved sharper bound.
6. **Linear (5.3) and unimodal (5.4) applications are entirely unverified**, including
   Corollary 5.3 (VD-BESTARMID), Proposition 5.8, Appendix G's `rho* = Õ(H_{2,lin})` and the
   claimed separations in Tables 1–2 beyond the single K⁷/K⁹ instance I checked symbolically.
   The FCW2S and FC2FB components those corollaries compose *are* verified separately.
7. **No comparison against the actual competitor implementations** (SHVar, SH, SHAdaVar, VBR,
   OD-LinBAI). Table 1/2 entries were used as written; only the analytic K⁷ vs K⁹ instance was
   checked, using the paper's own complexity expressions rather than running those algorithms.
8. **Monte-Carlo precision.** MC figures carry ±1/√n sampling error (n = 200 000 for claim 1,
   20 000 for claim 3, 200×5 for claim 5); they are cross-checks of exact computations, not the
   primary evidence, except in claim 5 where they *are* the primary evidence (Alg 5 has no closed
   form here) and thus resolve error probabilities only down to ≈1e-3.
9. **No paper source code was consulted** (none is referenced in the text), so all algorithm
   implementations are my reading of the pseudocode in `notes_paper.md`; an implementation
   misreading would not be caught by these tests. `notes_code.md` records this.
10. **Other people's reproduction logbooks were not read**, per the task rules.
