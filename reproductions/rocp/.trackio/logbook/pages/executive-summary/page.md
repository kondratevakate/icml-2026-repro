# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f27d33972b62", "created_at": "2026-07-28T15:58:37+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-28T15:58:37+00:00"}
-->
CPU-only proof, implementation, and cached-artifact reproduction of ROCP.
**Claim 1 is verified:** the closed-form fixed-set robust risk matches
independently solved primal linear programs in 28,950 comparisons with maximum
error `1.42e-14`. **Claim 3 is verified:** the paper-faithful `(n+1)`
Algorithm 1 passes 14,850 exhaustive exchangeability checks and empirical
miscoverage tracks alpha over 60 fresh run units. **Claim 4 is verified:**
ROCP lowers critical medical and driving mistakes, with zero observed medical
critical mistakes under the high-cost loss matrix. The audit also discloses
two paper/code divergences and small BDD reversals against a stronger
all-baselines/all-alpha statement.

## Scope & cost

| Item | Value |
| --- | --- |
| GPU / compute | CPU, NumPy/SciPy |
| Wall time | ~60 minutes for audits and 60 empirical run units |
| Feasibility | Complete locally; no GPU or protected raw data required |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_564249948727", "created_at": "2026-07-28T15:58:37+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-28T15:58:37+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 6px;font-size:18px;border-bottom:2px solid #1565c0;padding-bottom:8px">ROCP: fixed-set minimax policy and conformal coverage audit</h2>
  <p style="font-size:12px;color:#555">ICML 2026 · OpenReview VAXW59dyfk · CPU-only independent implementation</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:5px;text-align:left">Claim</th><th>Result</th><th>Verdict</th></tr>
    <tr><td style="padding:5px">C1 fixed-set minimax policy</td><td style="text-align:center">28,950 LP checks; max error 1.42e-14</td><td style="text-align:center;color:#27823b">VERIFIED</td></tr>
    <tr><td style="padding:5px">C3 finite-sample coverage</td><td style="text-align:center">14,850 orbit checks; 60 empirical units</td><td style="text-align:center;color:#27823b">VERIFIED</td></tr>
    <tr><td style="padding:5px">C4 critical mistakes</td><td style="text-align:center">20x2 COVID seeds; 20 BDD splits</td><td style="text-align:center;color:#27823b">VERIFIED</td></tr>
    <tr><td style="padding:5px">Official implementation</td><td style="text-align:center">2 reproducible paper/code divergences</td><td style="text-align:center">DISCLOSED</td></tr>
  </table>
</div>
````
