## Claim 5 — — inconclusive

**Verdict:** app. f/g, fig. 4(v2)

*Source:* App. F (interval construction) + Fig. 4 (v2); task cites App. G Fig. 6. *Script:* `verify_claim5.py` (9 min; 1,000 trials/config instead of 10,000, 50 comparisons per rater, R ∈ {5, 20, 50}).

**Finding: the paper's stated interval constant is inconsistent with its stated confidence level.** Appendix F gives `p99 = sqrt(diag(cov)) · 3.29`, but `sqrt(2)·erfcinv(0.001) = 3.2905` is the **99.9%** two-sided constant; the 99% one is 2.576. Mean Type I error over R ∈ {5,20,50}:

| rule | BBQ | Bayes-BT | Crowd-BT |
|---|---|---|---|
| paper's literal non-overlap w/ 3.29 | 0.00% | 0.00% | 0.07% |
| non-overlap w/ 2.576 | 0.03% | 0.00% | 1.13% |
| 99% interval on the Elo **difference** | 1.60% | 1.13% | 1.13% |

- Under the standard difference test, BBQ and Crowd-BT are indeed ≈1% ✔ — but **Bayes-BT is also ≈1%, not the claimed ≈0.1%**. Under the paper's literal recipe *everything* is ≈0.0–0.1%. No single decision rule reproduces the claimed 1% / 1% / 0.1% pattern simultaneously → inconclusive.
- The qualitative "Bayes-BT is the most conservative" ordering does hold under every rule.
- **Mutation:** with a true win probability of 0.60 (H₀ false), rejection rates jump to 97–100% for all models and rules. ✔ the ~1% is genuinely a null-hypothesis quantity, and the tests have power.
- *Caveat:* the credible-interval construction for BBQ/Bayes-BT (conditional Gamma posterior at the EM fixed point) is ours; the paper does not specify it fully, so the exact Bayes-BT number is implementation-sensitive.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 \u2014 inconclusive"}\n-->
