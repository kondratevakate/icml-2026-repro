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

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 Theorem 5.8 (VCG/SPA non-monotonicity for tCPA, 6.2%)"}\n-->
