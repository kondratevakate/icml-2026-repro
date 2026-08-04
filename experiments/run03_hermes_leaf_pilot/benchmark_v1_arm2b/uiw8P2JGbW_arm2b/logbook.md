# Reproduction logbook — uiw8P2JGbW (arm2)

**Paper:** *Model Monotonicity in Autobidding Auctions: When Do Better Predictions Lead to Better Outcomes?* (ICML 2026, Theory→Game Theory, A. Badanidiyuru)
**Arm:** arm2 — Hermes + K-Dense skill set, **no context compaction**
**Model:** tencent/hy3:free via localhost:8319/v1 · **CPU-only** (numpy 2.5.1 / scipy 1.18.0 / sympy 1.14.0)
**Base seed:** 20260803 (per-claim seed = base + claim index)

Paper PDF could not be downloaded (OpenReview `/pdf` endpoint is JS-gated; the fetch returned HTML). All work is therefore a **first-principles reconstruction** of the framework from the abstract + anchored claim statements, implemented in `common.py`:
queries carry true conversion rates `p_i(q)`; a model induces a partition; predictions are calibrated conditional means; refinement = splitting clusters (a mean-preserving spread); tCPA bidder `i` bids `mu_i · t_i · hatp_i(C)` (uniform bidding).

## Summary

| # | Claim (source) | Verdict |
|---|---|---|
| 1 | FPA + tCPA, no budgets, mu=1: refinement never decreases revenue (Theorem 5.1) | **verified** |
| 2 | Proof mechanism: convexity of f(p)=max_i t_i p_i + Jensen on a mean-preserving spread (Thm 5.1 / Sec. 5) | **verified** |
| 3 | mu=1 is conversion-, revenue- and welfare-maximizing for tCPA in FPA; welfare monotone when v=t (Thm 5.2, Cor. 5.3) | **verified** |
| 4 | Second-price/VCG can be non-monotone in revenue AND welfare simultaneously (Theorem 5.8) | **verified** |
| 5 | Budgets break revenue monotonicity even in FPA (Theorem 5.10) | **verified** |
| 6 | LP benchmark guarantees welfare monotonicity via lifting + Table 1 lists exactly three monotone settings (Thm 5.11, Table 1) | **inconclusive** |

## Claim 1 — Theorem 5.1 — **verified**
`verify_claim1.py` → `results/claim1.json`. 20,000 random refined/coarse instance pairs (3 bidders, 4 coarse clusters, 2–4 subclusters each).
- Violations of `Rev(fine) ≥ Rev(coarse)`: **0** (min gap −8.9e-16 = float noise); mean gain 0.2239; strict gain in 98.1% of instances.
- **Mutation A** (break calibration — coarse predictions multiplied by U[0.7,1.3]): 3,964 violations.
- **Mutation B** (swap convex `max` for concave `min` aggregator): 19,626 violations.
Both mutations break the property, so the test is not vacuous.

## Claim 2 — Theorem 5.1 proof / Section 5 — **verified**
`verify_claim2.py` → `results/claim2.json`.
- Convexity of `f(p)=max_i t_i p_i`: 0 violations in 200,000 random convex-combination checks; symbolic 4-region witness for n=2 passes.
- Exact accounting: `Rev(fine) − Rev(coarse)` equals the summed per-cluster Jensen gap `Σ_C m_C (E[f|C] − f(E[p|C]))` to **1.6e-15**; mean-preservation error **3.3e-16** — refinement is exactly a mean-preserving spread.
- Jensen gap negative in 0/5,000 instances.
- **Mutation** (concave `min` aggregator): gap is strictly *negative* in 4,898/5,000 instances and never positive — convexity is the load-bearing ingredient, exactly as claimed.

## Claim 3 — Theorem 5.2 / Corollary 5.3 — **verified**
`verify_claim3.py` → `results/claim3.json`. 2,000 instances × 300 random multiplier profiles + all one-bidder unilateral deviations.
- tCPA feasibility ⟺ `mu_i ≤ 1`: **0** mismatches (analytically `spend_i = mu_i t_i conv_i`).
- Feasible profiles beating mu=1 on revenue: **0**; on welfare (v=t): **0**; unilateral deviations beating mu=1 on the deviator's own conversions: **0**.
- Corollary 5.3 welfare-monotonicity violations under refinement: **0**.
- **Mutation** (v_i ≠ t_i, drawn as t·U[0.3,3]): the mu=1 allocation is *not* welfare-optimal in 1,732/2,000 instances — the `v = t` hypothesis is necessary.

## Claim 4 — Theorem 5.8 — **verified**
`verify_claim4.py` → `results/claim4.json`. Second-price with tCPA bidders raising `mu` to the largest tCPA-feasible value (damped iterated best response, bisection to a fixed point); values `v = t`.
- 150 instances tested; **22** show a *simultaneous* revenue and welfare drop under refinement. Best (after hill-climb): revenue −73.96%, welfare −70.42%. Random instances land near the paper's figure (closest sampled pair: revenue −9.5%, welfare −2.4%).
- **Mutation** (same instances run under first-price with mu=1): **0** revenue drops — non-monotonicity is specific to the second-price payment rule, as claimed.
- *Caveat:* the paper's exact 6.2%/6.2% construction is not recoverable without the PDF; the qualitative existence claim and its mechanism are reproduced, the specific pair of magnitudes is not matched.

## Claim 5 — Theorem 5.10 — **verified**
`verify_claim5.py` → `results/claim5.json`. FPA, uniform bidding, each bidder both tCPA (`mu ≤ 1`) and budget-constrained; equilibrium multipliers by damped iterated best response.
- 150 instances; **93** exhibit a revenue drop under refinement; max loss 99.85%; the closest sampled loss is **16.82%**, matching the paper's reported **16.8%** to two significant figures (coincidental instance, not the paper's construction).
- Equilibrium multipliers do shift downward on the refined partition (recorded in `best_instance.mu_fine` / `mu_coarse`), which is the stated mechanism.
- **Mutation** (budgets set to +∞): **0** revenue drops — recovers Theorem 5.1, confirming budgets are the cause.

## Claim 6 — Theorem 5.11 + Table 1 — **inconclusive**
`verify_claim6.py` → `results/claim6.json`. Centralized non-strategic LP benchmark solved with `scipy.optimize.linprog` (HiGHS): maximize `Σ m_C v_i hatp_i(C) x_{C,i}` s.t. one slot per cluster and per-bidder budgets on `Σ m_C t_i hatp_i(C) x`.
- Welfare monotonicity violations across 400 instances: **0**.
- Explicit **lifting** check (copy the coarse solution to every subcluster): feasible in 400/400; objective error **4.4e-16**; budget overshoot **3.3e-16** — the lift is exactly value- and feasibility-preserving, which is the theorem's proof device.
- **Mutation** (miscalibrated coarse predictions): 19 monotonicity violations — calibration is required.
- **Why inconclusive:** the claim is a conjunction, and its second conjunct ("Table 1 identifies exactly three settings where monotonicity holds") requires the paper's table, which could not be retrieved. The Theorem 5.11 half is reproduced cleanly; the completeness half is untested.

## Threats to validity
- No access to the paper body: the auction/bidding model, the second-price equilibrium selection, and the LP benchmark's cost model are reconstructions consistent with the claim statements and the standard autobidding literature. Exact counterexample instances (6.2%, 16.8%) are therefore not byte-identical reproductions.
- Single-slot auctions only; multi-slot and max-CPA bidder types were out of scope for the anchored claims.
- Equilibrium computation for claims 4–5 is a damped iterated best response with finite bisection precision (1e-3-ish on `mu`), so reported magnitudes carry that tolerance.

## Reproduce
```bash
python3 -m venv venv && ./venv/bin/pip install numpy scipy sympy
for n in 1 2 3 4 5 6; do ./venv/bin/python verify_claim$n.py; done   # ~15 min total, CPU only
```
