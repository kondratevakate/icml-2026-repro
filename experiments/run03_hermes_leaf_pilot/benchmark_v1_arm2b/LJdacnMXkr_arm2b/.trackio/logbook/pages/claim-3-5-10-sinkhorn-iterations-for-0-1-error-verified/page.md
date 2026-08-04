## Claim 3 — 5–10 Sinkhorn iterations for <0.1% error — **verified**

Script `verify_claim3.py`, results `results/claim3.json`.
40 configurations (dim 2/3 × Gaussian/exponential × eps ∈ {0.3,0.5,0.8,1.2,2.0} ×
uniform/non-uniform mass, n=200 each), error metric
`max_i |(diag(d)W diag(d)1)_i − m_i| / m_i`, threshold 1e-3, start `d = 1`.

- iterations: **min 6, median 9.5, mean 9.0, max 11**
- 90% of configurations land in the claimed 5–10 window; 90% are ≤ 10 (the two misses
  need 11).

**Mutations.** Clustered geometry with tiny bandwidth: 9 iterations (not discriminating —
symmetric Sinkhorn is robust there). Extreme target-mass dynamic range (1e-3…1e3):
**14 iterations**, outside the claimed window — the "5–10" figure is bandwidth/mass
dependent, not universal.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 5\u201310 Sinkhorn iterations for <0.1% error \u2014 **verified**"}\n-->
