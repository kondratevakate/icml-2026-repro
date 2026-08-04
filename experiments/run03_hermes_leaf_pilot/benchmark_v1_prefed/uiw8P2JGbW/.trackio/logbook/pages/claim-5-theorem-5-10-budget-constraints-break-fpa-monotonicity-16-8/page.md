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

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Theorem 5.10 (Budget constraints break FPA monotonicity, 16.8%)"}\n-->
