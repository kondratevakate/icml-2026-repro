# Claim 4 — Table 4 (traffic LWR) — inconclusive

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_77a4f37360cb", "created_at": "2026-08-02T05:00:00+00:00", "title": "Claim 4 \u2014 Table 4 (traffic LWR) \u2014 inconclusive"}
-->
**Paper source:** Section 4.3, Table 4.
**Record:** `results/claim4.json`. Same data/training blockers as claim 3.
**Discrepancy recorded:** the anchored claim quotes an upper-bound violation rate of **1.87 %**
for FluxNet-D, but Table 4 as printed reports **2.87 %** (ResNet backbone) and **2.01 %** (FNO
backbone); 1.87 % matches neither, and the nearest printed value is FNO-AR's V_ub = 1.88 %.
Unresolvable without the authors' raw numbers. The structural sub-parts (machine-precision
conservation; D-head having no strict dual-bound guarantee but near-zero empirical violations)
are covered by claims 1 and 6.
