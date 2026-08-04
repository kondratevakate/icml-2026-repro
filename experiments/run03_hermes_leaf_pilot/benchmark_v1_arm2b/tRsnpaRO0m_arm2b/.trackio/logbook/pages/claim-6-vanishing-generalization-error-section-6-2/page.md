## Claim 6 — vanishing generalization error (Section 6.2)

Script `verify_claim6.py` → `results/claim6.json`. Verdict: **verified** (with an explicit caveat).
Mutation test: yes.

Fixed capacity (width 48), sweep training size `m`, 6 replicates, label noise σ = 0.05, gap =
`|test MSE − train MSE|`:

| m | 25 | 50 | 100 | 200 | 400 | log-log slope |
|---|---|---|---|---|---|---|
| bofop gap | 2.78e-3 | 1.34e-3 | 5.33e-4 | 4.98e-4 | **2.23e-4** | **−0.871** |
| non-bofop gap (mutation) | 4.61e-2 | 1.56e-2 | 7.78e-3 | 2.69e-3 | 1.03e-3 | −1.350 |

The gap vanishes with `m` at a rate comfortably at least the `m^{-1/2}` that a covering-number
argument over a compact, uniformly equicontinuous class guarantees.

**Pitfall handled:** with a *noiseless* target the ridge head fits exactly, the gap collapses to the
float64 noise floor (~1e-7) and the fitted rate is **vacuous, not failing**. σ = 0.05 label noise is
therefore added.

**CAVEAT (kept visible, not tuned away).** The non-bofop mutation is reported as **two separate
named checks** in the JSON:
- `mutation_raises_gap_level_at_every_m: true` — the unbounded-fiber family has a 2–21× larger gap at
  every single `m`, so the fiber bound demonstrably improves the constant;
- `mutation_degrades_asymptotic_rate: false` — its slope is *steeper* (−1.35 vs −0.871), so this
  experiment does **not** isolate compactness as strictly *necessary* for the asymptotic rate.

The failing sub-check is left in the artifact rather than deleted. A `verified` with a named caveat
is worth more than a `verified` that hides one.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 vanishing generalization error (Section 6.2)"}\n-->
