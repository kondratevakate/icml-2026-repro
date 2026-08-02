# Claim 5 — Table 5 (spinodal decomposition, 17.3× speedup) — inconclusive

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7f802876cc99", "created_at": "2026-08-02T04:00:00+00:00", "title": "Claim 5 \u2014 Table 5 (spinodal decomposition, 17.3\u00d7 speedup) \u2014 inconclusive"}
-->
**Paper source:** Section 4.4, Table 5 and Figure 3.
**Record:** `results/claim5.json`.
The 17.3× speedup is defined *relative to a GPU-accelerated explicit solver*; with no GPU in this
environment the reference timing cannot be measured and the ratio is not defined on CPU.
It also requires three trained FluxNet-D models (r = 3/5/9) on unreleased 50 000-step
Cahn–Hilliard trajectories. The two-point correlation metric S̄₂(r) is itself implementable via
FFT autocorrelation, but there is no trained model to evaluate it on. No toy substitute.
