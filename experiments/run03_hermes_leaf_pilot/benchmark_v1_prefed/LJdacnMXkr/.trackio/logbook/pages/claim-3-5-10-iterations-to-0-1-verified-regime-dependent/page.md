## Claim 3 — 5–10 iterations to 0.1 % — **verified (regime-dependent)**
`verify_claim3.py` → `results/claim3.json`

Sweep of 48 configurations: {Gaussian, exponential} × d∈{2,3} × n∈{100,300} × ε∈{0.15,0.3,0.6}
× {uniform, clustered} sampling. Error metric `maxᵢ|(P1)ᵢ − 1|`.

- iterations to < 1e-3: **min 7, median 8, p90 9, max 9** — 100 % within 5–10, 0 % within 5
- worst error after exactly 10 iterations: 3.6e-4

So the *upper* end of the claimed range is reproduced exactly; the "as few as 5" end was never
reached in our sweep (7 was the minimum).

**Mutation.** Extreme mass imbalance (mass ratio ~1e-6) on an elongated chain domain (n = 400,
length 200, ε = 0.45): **14 iterations**, error still 1.1 % after 10 — outside the claimed range.
The claim is an empirical statement about well-conditioned kernels, not a bound.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 5\u201310 iterations to 0.1 % \u2014 **verified (regime-dependent)**"}\n-->
