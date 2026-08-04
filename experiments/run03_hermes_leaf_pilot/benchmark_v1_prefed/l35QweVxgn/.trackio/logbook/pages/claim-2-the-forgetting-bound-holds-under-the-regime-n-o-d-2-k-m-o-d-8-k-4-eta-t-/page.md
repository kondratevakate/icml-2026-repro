## Claim 2 — — The forgetting bound holds under the regime n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2) on a d-dim XOR-cluster dataset with K tasks (Theorem 1).

- **Verdict:** `verified`
- **Source:** Theorem 2.1 regime conditions (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)), arXiv:2510.05573v2.
- **Seed:** 20261004  (master 20260802)
- **Mutation test:** Break ONE factor: (a) width m=d^2 (below d^8 K^4), (b) sample size n=d (below d^2 K).  [mutation breaks]  With correct m but n=d, term1 stays order-1 and grows; with correct n but m=d^2, term3 grows as d^3. Neither factor alone drives the bound to zero -- confirming 'joint control'.
- **Key numerics:**
  - `term2_by_d`: {"16": 0.6247052776618358, "32": 0.4997642221294687, "64": 0.4164701851078906, "128": 0.3569744443781919}
  - `term3_by_d`: {"16": 0.13008556131285048, "32": 0.08325475924022432, "64": 0.05781580502793356, "128": 0.04247691797970629}
  - `doubling_ratios`: {"term1": 0.8451542547285165, "term2": 0.7142857142857143, "term3": 0.5102040816326531}
  - `empirical_fluctuation_by_d`: {"16": 3.410632821818748e-07, "32": 1.7352943060102343e-08, "64": 2.1069328829309833e-09, "128": 8.983466520504458e-11}
- **Note on vanishing:** Under the regime every term vanishes as d->inf (poly-logarithmically): term1 ~ 1/sqrt(log d) (via the polylog hidden in n=O~(d^2 K)), term2 ~ 1/log d, term3 ~ 1/log^2 d. Hence F_tr = o_d(1) as claimed.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 \u2014 The forgetting bound holds under the regime n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2) on a d-dim XOR-cluster dataset with K tasks (Theorem 1)."}\n-->
