# Reproduction task — Model Monotonicity in Autobidding Auctions: When Do Better Predictions Lead to Better Outcomes?

OpenReview: https://openreview.net/forum?id=uiw8P2JGbW
Area: Theory
Anchored claims (6):

1. For target-CPA bidders without budget constraints using uniform bidding with multiplier mu=1 in first-price auctions, refining the prediction model (partitioning clusters into subclusters while preserving calibration) never decreases platform revenue, i.e., Rev(M_A) >= Rev(M_B) when M_A refines M_B (Theorem 5.1).

2. The first-price auction revenue monotonicity proof for tCPA bidders relies on convexity of f(p) = max_i(t_i · p_i) via Jensen's inequality, since model refinement corresponds to a mean-preserving spread of this convex function (Theorem 5.1, Section 5).

3. Uniform bidding at multiplier mu=1 is simultaneously conversion-maximizing, revenue-maximizing among equilibria, and welfare-maximizing for tCPA bidders in first-price auctions (Theorem 5.2), and welfare monotonicity follows as a corollary when per-conversion values equal targets (Corollary 5.3).

4. There exist instances with tCPA bidders where second-price (VCG) auctions exhibit non-monotonicity in both revenue and welfare under model refinement, with a constructed counterexample showing a simultaneous 6.2% decrease in both metrics (Theorem 5.8).

5. Introducing budget constraints breaks revenue monotonicity even in first-price auctions, as optimal bidder multipliers shift under refinement and reduce competitive pressure, exemplified by a counterexample with a 16.8% revenue loss (Theorem 5.10).

6. A centralized, non-strategic linear-programming benchmark allocation guarantees welfare monotonicity under budget constraints via a 'lifting' construction that preserves feasibility across refined partitions (Theorem 5.11), and the paper's overall characterization identifies exactly three settings where monotonicity holds (Table 1).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
