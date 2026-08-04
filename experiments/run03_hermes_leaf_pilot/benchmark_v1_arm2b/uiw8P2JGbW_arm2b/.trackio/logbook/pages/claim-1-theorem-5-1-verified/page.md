## Claim 1 — Theorem 5.1 — **verified**
`verify_claim1.py` → `results/claim1.json`. 20,000 random refined/coarse instance pairs (3 bidders, 4 coarse clusters, 2–4 subclusters each).
- Violations of `Rev(fine) ≥ Rev(coarse)`: **0** (min gap −8.9e-16 = float noise); mean gain 0.2239; strict gain in 98.1% of instances.
- **Mutation A** (break calibration — coarse predictions multiplied by U[0.7,1.3]): 3,964 violations.
- **Mutation B** (swap convex `max` for concave `min` aggregator): 19,626 violations.
Both mutations break the property, so the test is not vacuous.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 5.1 \u2014 **verified**"}\n-->
