# Reproduction plan — uiw8P2JGbW

**Paper:** "Model Monotonicity in Autobidding Auctions: When Do Better
Predictions Lead to Better Outcomes?" (arxiv 2605.31036, ICML 2026).

**Goal:** Reproduce the 6 anchored claims on CPU with numpy/scipy/sympy.
Each claim gets `verify_claim<N>.py` + `results/claim<N>.json` + a mutation
test; final summary in `logbook.md`; determinism check in
`check_reproducibility.py`.

**Environment:** Python 3.12 venv at `.venv/` (numpy 2.5.1, scipy 1.18.0,
sympy 1.14.0). No GPU, no ML. Pinned RNG seed = 20260501.

**Shared code:** `common.py` — model/partition construction, refinement check,
FPA revenue for tCPA (mu=1), SPA/VCG auction, FPA-with-budget auction, LP
allocation benchmark + lifting.

## Claims
1. **Theorem 5.1** — FPA revenue monotonicity for tCPA (mu=1, no budget):
   `Rev(M_A) >= Rev(M_B)` when `M_A` refines `M_B`.
   → Random stress test over 2000 refined model pairs + illustrative example.
2. **Theorem 5.1 proof mechanism** — `f(p)=max_i t_i p_i` is convex; refinement
   is a mean-preserving spread; Jensen gives the inequality.
   → Random Jensen checks + per-cluster MPS/Jensen verification.
3. **Theorem 5.2 / Corollary 5.3** — mu=1 is conversion-, revenue-, welfare-
   maximizing for tCPA in FPA; welfare monotonicity follows (v_i=t_i).
   → Revenue vs mu; welfare-monotonicity stress test.
4. **Theorem 5.8** — VCG/SPA for tCPA non-monotone in revenue AND welfare;
   explicit 6.2% counterexample (Appendix B.1).
   → Reconstruct B.1, run SPA, confirm 3.23→3.03 (≈6.2%).
5. **Theorem 5.10** — Budget constraints break FPA revenue monotonicity;
   16.8% counterexample (Appendix B.2).
   → Derive budget-binding alpha, run FPA, confirm 5.5268→4.5977 (16.8%).
6. **Theorem 5.11 / Table 1** — LP lifting guarantees monotonicity under
   budgets; exactly three monotonicity settings.
   → Solve LP for coarse/refined, lift coarse optimum, verify feasibility +
   objective; MAX-CPA VCG welfare stress; B.4 66% counterexample; Table 1.

## Mutation tests (every verified claim)
- C1: drop refinement / break calibration → monotonicity can fail.
- C2: non-convex f / broken MPS → Jensen fails.
- C3: mu<1 lowers revenue; v_i≠t_i breaks Welfare==Revenue.
- C4: same instance under FPA (mu=1) becomes monotone (VCG-specific).
- C5: remove budget → Theorem 5.1 regime (monotone again).
- C6: corrupt refined predictions → lifting infeasible (Theorem 5.11 fails).
