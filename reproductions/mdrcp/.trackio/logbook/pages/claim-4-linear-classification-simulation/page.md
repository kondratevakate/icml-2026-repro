# Claim 4: Linear classification simulation


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mdrcp_v2_c4", "created_at": "2026-07-29T08:10:00+00:00", "title": "Claim 4: Linear classification simulation"}
-->
**Anchored claim.** In the paper's Linear classification simulations, MDCP
sets are 34.39% smaller than Baseline-agg while retaining tight worst-source
coverage (Section 5.2, Figure 2).

**Verdict: PRE-VERDICT.**

The first reproduction obtained valid coverage but only a 4.73% reduction
under a simplified optimizer, so it was correctly judged inconclusive.
Official repository commit `6a7d5ba990c98aff3a9510bc40357254f3107e68` is now
pinned and its real Linear evaluator is being profiled. Do not promote this
claim until the full 100-trial panel completes and reproduces both coverage
and the efficiency direction.
