## Claim 4 — Theorem 5.8 — **verified**
`verify_claim4.py` → `results/claim4.json`. Second-price with tCPA bidders raising `mu` to the largest tCPA-feasible value (damped iterated best response, bisection to a fixed point); values `v = t`.
- 150 instances tested; **22** show a *simultaneous* revenue and welfare drop under refinement. Best (after hill-climb): revenue −73.96%, welfare −70.42%. Random instances land near the paper's figure (closest sampled pair: revenue −9.5%, welfare −2.4%).
- **Mutation** (same instances run under first-price with mu=1): **0** revenue drops — non-monotonicity is specific to the second-price payment rule, as claimed.
- *Caveat:* the paper's exact 6.2%/6.2% construction is not recoverable without the PDF; the qualitative existence claim and its mechanism are reproduced, the specific pair of magnitudes is not matched.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 Theorem 5.8 \u2014 **verified**"}\n-->
