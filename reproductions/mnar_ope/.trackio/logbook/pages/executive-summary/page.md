# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6b4350c6fc6c", "created_at": "2026-07-29T15:03:33+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T15:03:33+00:00"}
-->
**Outcome.** The five challenge claims were audited against
arXiv `2606.20206v1`, official ShadOPE commit
`4231ba5d46046c66c0efae6d58662bfca5087147`, an analytic counterexample, and
independent CPU diagnostics. C1 is **FALSIFIED as written** because the anchored
claim omits a necessary Picard regularity condition. C2-C5 are **VERIFIED**,
with the assumptions and two C4 proof-bookkeeping defects reported explicitly.

## Scope and cost

| Item | Value |
| --- | --- |
| GPU | None |
| Protected medical data | None |
| Independent simulation | 60,000 one-step samples |
| Unit tests | 5 passing |
| Prepared score | 10/10 |
| Conservative expected score | 8/10 |

The official CPU smoke path also ran all five estimators at `n=32`, `T=2`.
Cached official simulation and MIMIC-III outputs were inspected for feasibility
but were not counted as independent evidence.

Verdicts: `C1=FALSIFIED`, `C2=VERIFIED`,
`C3=VERIFIED`, `C4=VERIFIED`, `C5=VERIFIED`.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_6eaf701ecd26", "created_at": "2026-07-29T15:03:33+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T15:03:33+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #cfd7df;padding:20px;background:#fff;color:#17202a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">MNAR OPE: independent theorem and assumption audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | vpSFJoxyDz | arXiv 2606.20206v1 | ShadOPE 4231ba5</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 bridge existence</td><td>Complete wrapped-Gaussian operator; inverse L2 norm diverges</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">C2 relevance</td><td>60k-sample conditional diagnostic, residual r=0.253</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">C3 bridge error</td><td>2,000 finite inverse-problem reductions, 0 violations</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">C4 policy rate</td><td>Recurrence, K product, critical radii, T^2 composition</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">C5 no future dependence</td><td>AST dependency audit plus nested likelihood-ratio test</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
</table>
<p style="font-size:12px;margin:12px 0 0">Prepared 10/10; conservative expected 8/10. CPU-only, no protected medical data.</p>
</div>
````
