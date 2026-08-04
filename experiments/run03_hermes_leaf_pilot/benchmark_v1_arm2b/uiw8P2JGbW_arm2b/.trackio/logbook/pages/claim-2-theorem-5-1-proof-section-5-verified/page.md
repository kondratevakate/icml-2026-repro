## Claim 2 — Theorem 5.1 proof / Section 5 — **verified**
`verify_claim2.py` → `results/claim2.json`.
- Convexity of `f(p)=max_i t_i p_i`: 0 violations in 200,000 random convex-combination checks; symbolic 4-region witness for n=2 passes.
- Exact accounting: `Rev(fine) − Rev(coarse)` equals the summed per-cluster Jensen gap `Σ_C m_C (E[f|C] − f(E[p|C]))` to **1.6e-15**; mean-preservation error **3.3e-16** — refinement is exactly a mean-preserving spread.
- Jensen gap negative in 0/5,000 instances.
- **Mutation** (concave `min` aggregator): gap is strictly *negative* in 4,898/5,000 instances and never positive — convexity is the load-bearing ingredient, exactly as claimed.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Theorem 5.1 proof / Section 5 \u2014 **verified**"}\n-->
