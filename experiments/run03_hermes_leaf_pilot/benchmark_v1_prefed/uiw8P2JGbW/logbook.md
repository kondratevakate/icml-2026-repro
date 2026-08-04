# Logbook — Reproduction of uiw8P2JGbW

**Paper:** Model Monotonicity in Autobidding Auctions: When Do Better
Predictions Lead to Better Outcomes? (arxiv 2605.31036, ICML 2026)
**Area:** Theory · **Format:** CPU-only numpy/scipy/sympy (no GPU, no ML)
**Date:** 2026-08-02 · **Seed (pinned):** 20260501
**Workspace:** `uiw8P2JGbW/` (this directory) · venv `.venv/` (numpy 2.5.1,
scipy 1.18.0, sympy 1.14.0)

## Summary verdict table

| # | Anchored claim | Verdict | Source | Key reproduction |
|---|---------------|---------|--------|-----------------|
| 1 | FPA revenue monotonicity for tCPA (μ=1, no budget) | **verified** | Thm 5.1 / App. A.2 | Rev(M_A) ≥ Rev(M_B) on 2000/2000 refined pairs |
| 2 | Convexity of f + mean-preserving spread (proof mechanism) | **verified** | Thm 5.1 proof / App. A.2 | 0 Jensen violations; MPS & Jensen hold per cluster |
| 3 | μ=1 optimal & welfare-monotonicity corollary | **verified** | Thm 5.2 / Cor. 5.3 | Revenue ↑ in μ, max at 1.0; welfare mono 2000/2000 |
| 4 | VCG/SPA tCPA non-monotone, 6.2% both | **verified** | Thm 5.8 / App. B.1 | 3.23 → 3.03, drop 6.20% rev / 6.19% welf |
| 5 | Budget breaks FPA monotonicity, 16.8% | **verified** | Thm 5.10 / App. B.2 | 5.5268 → 4.5977, drop 16.81% |
| 6 | LP lifting monotonicity + Table 1 (3 settings) | **verified** | Thm 5.11 / App. B.5; Table 1 | Lift feasible+equal objective; 3 setting families |

**All 6 anchored claims reproduced and verified.** No GPU/network required;
every computation is from first principles in `common.py` + `verify_claim<N>.py`.
Full numeric detail is in `results/claim<N>.json`.

---

## Claim 1 — Theorem 5.1 (FPA revenue monotonicity for tCPA)

**What the paper claims:** for tCPA bidders without budget using uniform
bidding with μ=1 in a first-price auction, refining the prediction model never
decreases platform revenue: `Rev(M_A) ≥ Rev(M_B)` whenever `M_A` refines `M_B`.
Revenue has closed form `Rev(M) = Σ_C w_C · max_i t_i·p_i,C`.

**Reproduction.** Built 2000 random calibrated model pairs (coarse partition,
refined by splitting each cluster), computed revenue from first principles,
confirmed `Rev(M_A) ≥ Rev(M_B)` in **every** instance (min margin ≈ 0, never
negative within 1e-9). An illustrative example is recorded in the JSON.

**Mutation test (property must break).** (a) Constructed `M_A` *not*
refining `M_B` → revenue can strictly decrease (observed). (b) Replaced
calibrated predictions with random values (calibration broken) → the Jensen
step is invalid and revenue can decrease (observed). So both theorem conditions
(refinement + calibration preservation) are essential.

**Verdict: verified.** Source: Theorem 5.1, §5.2.1, proof Appendix A.2.

---

## Claim 2 — Proof mechanism (convexity + mean-preserving spread)

**What the paper claims:** the proof relies on convexity of
`f(p)=max_i(t_i·p_i)` (pointwise max of linear functions) via Jensen's
inequality, because refinement is a mean-preserving spread of this convex
function.

**Reproduction.** (1) Random Jensen tests on `f` over 5000 points: **0**
convexity violations. (2) Per coarse cluster, verified the sub-cluster
predictions form a distribution whose mean equals the coarse prediction
(calibration / mean-preserving spread): **0** failures. (3) Jensen per cluster
`Σ_j λ_j f(p_sub_j) ≥ f(Σ_j λ_j p_sub_j) = f(p_coarse)`: **0** failures, min
margin ≈ 0.

**Mutation test.** (a) Replaced `f` by the non-convex function `min_i(t_i·p_i)`
→ Jensen fails (observed). (b) Broke the mean-preserving spread (sub
predictions no longer average to coarse) → the Jensen inequality fails
(observed). So convexity *and* calibration are both required.

**Verdict: verified.** Source: Theorem 5.1 proof, §5.2.1 / Appendix A.2.

---

## Claim 3 — Theorem 5.2 / Corollary 5.3 (μ=1 optimality + welfare)

**What the paper claims:** uniform bidding at μ=1 is simultaneously
conversion-maximizing, revenue-maximizing among equilibria, and
welfare-maximizing for tCPA bidders in FPA. Welfare monotonicity follows as a
corollary (Assumption 3.1: v_i = t_i ⇒ welfare = revenue for tCPA).

**Reproduction.** (1) Feasibility: CPA per won cluster = μ·t_i, so μ≤1
required; at μ=1 the CPA constraint binds. (2) Revenue `Rev(μ)=μ·Σ_C w_C
max_i t_i·p_i,C` is monotone increasing in μ and maximized at the feasible
boundary μ=1 (mean revenue over 1000 instances: μ=0.3→0.648, 0.5→1.080,
0.7→1.512, 0.9→1.944, 1.0→2.159). (3) Welfare at μ=1 equals
`Σ_C w_C max_i t_i·p_i,C`, i.e. the allocation `argmax_i t_i·p_i,C` which is
welfare-maximizing (value = t_i). (4) Welfare-monotonicity corollary: over 2000
refined model pairs, `Welfare(M_A) ≥ Welfare(M_B)` in **every** instance
(follows because welfare == revenue for tCPA).

**Mutation test.** (a) μ=0.5 yields strictly lower revenue than μ=1
(property shifts). (b) Setting v_i ≠ t_i breaks the Welfare == Revenue identity
(observed in 1000/1000 instances), so the Corollary-5.3 premise fails.

**Verdict: verified.** Source: Theorem 5.2 and Corollary 5.3, §5.2.1.

---

## Claim 4 — Theorem 5.8 (VCG/SPA non-monotonicity for tCPA, 6.2%)

**What the paper claims:** there exist tCPA-bidder instances where VCG
(=SPA for single items) is non-monotone in **both** revenue and welfare under
model refinement, with a constructed counterexample showing a simultaneous
~6.2% decrease in both.

**Reproduction (counterexample B.1, reconstructed from first principles).**
Two tCPA bidders A (t=10), B (t=1); four impressions; coarse partition
{{0,1},{2,3}}, fine = singletons; canonical CPA-binding equilibrium
multipliers. Ran the SPA auction per impression:

- Coarse: Revenue = 3.23, Welfare = 3.23 (both bidders exactly at tCPA;
  allocations A wins {0,1}, B wins {2,3}).
- Fine: Revenue = 3.03, Welfare = 3.03 (A wins {0}, B wins {1,2,3}).
- Decrease: **6.20%** revenue, **6.19%** welfare (paper: 6.2% both).

Refinement `M_A ≥ M_B` holds (singletons refine any partition). CPA binds for
both bidders under the published multipliers.

**Mutation test (property must break/shift).** Ran the *same* instance under
FPA (μ=1 uniform bidding). By Theorem 5.1 FPA revenue is monotone: coarse 3.23
→ fine 3.30 (revenue **increases** by −2.17%, i.e. no decrease). This confirms
the non-monotonicity in Theorem 5.8 is driven by VCG's second-price (externality)
payment, not by the model refinement itself. (The earlier "fixed multipliers"
idea was dropped: Jensen monotonicity holds for FPA, not VCG.)

**Verdict: verified** (the existence of a 6.2%-drop counterexample is
reproduced). Source: Theorem 5.8, §5.3.1 / Appendix B.1.

---

## Claim 5 — Theorem 5.10 (Budget constraints break FPA monotonicity, 16.8%)

**What the paper claims:** introducing budget constraints breaks FPA revenue
monotonicity even for tCPA bidders; optimal multipliers shift under refinement
and reduce competitive pressure, exemplified by a 16.8% revenue loss.

**Reproduction (counterexample B.2, reconstructed from first principles).**
Two advertisers, FPA, uniform bidding `b_i(u)=α_i·p^_i,u`; Adv1 budget B1=3.185
(binding), t1=8.674 (non-binding); Adv2 budget ∞, t2=1.662; coarse partition
{1,2},{3,4}, fine = singletons. **Derived the budget-binding α1 from each
allocation** (first principles, not copied rounded constants):
α1_coarse = 3.185/1.115 = **2.8565**, α1_fine = 3.185/1.631 = **1.9528**
(exactly matching the paper). Verified self-consistency (derived α1 reproduces
the assumed winning sets) and that Adv1's budget binds in both models.

- Coarse: Revenue = 5.5268 (Adv2 wins {1,2}, Adv1 wins {3,4}).
- Fine: Revenue = 4.5977 (Adv1 wins {1,3,4}, Adv2 wins {2}).
- Decrease: **16.81%** (paper: 16.8%).

**Mutation test.** Removed the budget (B1=∞) → reverts to the Theorem 5.1
regime. With the optimal no-budget multiplier (α_i = t_i), revenue becomes
monotone again: fine revenue **exceeds** coarse (change −8.19%, i.e. no
decrease), confirming the budget constraint is the sole cause.

**Verdict: verified** (the 16.8% revenue-loss counterexample is reproduced).
Source: Theorem 5.10, §5.3.2 / Appendix B.2.

---

## Claim 6 — Theorem 5.11 (LP lifting) + Table 1 (three settings)

**What the paper claims:** a centralized, non-strategic LP allocation benchmark
guarantees welfare monotonicity under budget constraints via a *lifting*
construction that preserves feasibility across refined partitions (Theorem
5.11); the overall characterization identifies **exactly three settings** where
monotonicity holds (Table 1).

**Reproduction of the lifting (Theorem 5.11).** Built calibrated coarse+refined
model pairs with budgets; solved the allocation LP
`max Σ w_C x_{i,C}·a_i·p_i,C` s.t. budget `Σ w_C x_{i,C}·a_i·p_i,C ≤ B_i` and
supply `Σ_i x_{i,C} ≤ 1`. Then **lifted** the coarse optimum to the refined
partition (copy allocation fractions to every sub-cluster). Over 50 instances:
the lifted solution was **feasible** for the refined LP (supply + budget) and
achieved the **same objective** → therefore `Rev_LP(M_A) ≥ Rev_LP(M_B)` (tCPA
surrogate revenue) and `Welfare_LP(M_A) ≥ Welfare_LP(M_B)`. Directly solving
both LPs also confirmed the inequality in **every** instance. The argument is
identical for MAX-CPA (a_i = v_i).

**The three monotonicity settings (Table 1).** The paper's "exactly three
settings" are three setting *families*, each reproduced here:
1. **tCPA FPA, no budget** (revenue + welfare) — Claim 1.
2. **MAX-CPA VCG, no budget** (welfare) — Theorem 5.5; stress-tested here over
   2000 instances with **0** failures (same Jensen structure as Claim 1 with
   v_i for t_i).
3. **LP allocation, with budgets** (welfare; tCPA surrogate revenue) — this
   claim (Theorem 5.11).

The remaining Table-1 rows are non-monotonic; two are reproduced as explicit
counterexamples: Claim 4 (tCPA VCG, 6.2%) and Claim 5 (tCPA+budget FPA, 16.8%).
As supporting evidence for the MAX-CPA FPA negative row, the Appendix B.4
counterexample was also reproduced: MAX-CPA FPA with designated multiplier
profiles gives Revenue 0.68 → 0.23, a **66.18%** loss (paper: 66%).

**Mutation test.** Corrupted the refined predictions so they no longer average
to the coarse prediction (calibration broken). The lifted coarse solution then
violates the refined budget constraints → the Theorem 5.11 monotonicity
guarantee fails. So calibration preservation (central to the lifting) is
required.

**Verdict: verified** (LP lifting reproduced; Table 1's three settings each
substantiated). Source: Theorem 5.11, §5.3.3 / Appendix B.5; Table 1.

---

## Reproducibility & honesty notes

- **Determinism:** `check_reproducibility.py` re-runs every claim with the
  pinned seed 20260501 and asserts the verdicts and key numbers match
  `results/claim<N>.json`. (All randomness via `np.random.default_rng(SEED)`.)
- **Honest verdicts:** every claim is `verified` because each was reproduced
  from first principles to the paper's stated numbers (6.2%, 16.8%, 66%, and
  the general monotonicity inequalities over thousands of random instances).
  No claim was marked `falsified` or `inconclusive` because no discrepancy with
  the paper was found.
- **Caveats:** the non-monotonicity counterexamples (Claims 4, 5, and B.4)
  use the paper's *published canonical equilibrium multiplier profiles*; we
  verified their internal consistency (CPA binds, allocation matches, refinement
  holds) and recomputed the auction outcomes rather than merely copying the
  printed totals. Mutation tests confirm the results are sensitive to the
  setup, as expected for existence-style claims.
- **Environment:** Python 3.12 venv created in `.venv/` (numpy/scipy/sympy);
  no external data or network access needed for the computations.

## Files produced
- `common.py` — shared model/auction/LP primitives.
- `verify_claim1.py` … `verify_claim6.py` — per-claim computations.
- `results/claim1.json` … `results/claim6.json` — numeric results + mutation tests.
- `check_reproducibility.py` — determinism checker (pinned seed).
- `plan.md`, `logbook.md` — plan and this logbook.
