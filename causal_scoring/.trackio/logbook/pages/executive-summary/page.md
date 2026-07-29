# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_712038cb2f74", "created_at": "2026-07-29T19:11:25+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T19:11:25+00:00"}
-->
**Outcome.** This CPU-only reproduction independently audits
six claims using exact algebra, high-precision grids, destructive mutations,
and a 30-seed 10-fold synthetic panel.

| Claim | Result | Score |
| --- | --- | ---: |
| IPW MSE bound | Partial: fixed-propensity scope | 1/2 |
| Task curvature | Verified | 2/2 |
| Strictly proper loss | Verified | 2/2 |
| Quartic mapping | Partial: no-vanish clause false | 1/2 |
| Kang-Schafer mechanism | Partial independent simulation | 1/2 |
| Broad benchmarks | Inconclusive | 0/2 |
| **Prepared** |  | **7/12** |

No leaderboard or peer reproduction is used as evidence.

Provenance: [arXiv](https://arxiv.org/abs/2606.03332) and
[OpenReview](https://openreview.net/forum?id=JTwryHNicJ).


---
<!-- trackio-cell
{"type": "figure", "id": "cell_a87baa73d8e6", "created_at": "2026-07-29T19:11:25+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T19:11:25+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">Tailored causal scoring rules: independent audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | JTwryHNicJ | arXiv 2606.03332v1 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">IPW bound</td><td>Closes conditionally; cross-fold dependence omitted</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Curvature</td><td>Exact symbolic identity plus mutation</td><td style="color:#067647;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Proper loss</td><td>Exact derivatives and exhaustive risk minima</td><td style="color:#067647;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Canonical map</td><td>Quartic holds; no-vanish clause false</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Kang-Schafer</td><td>7.10x RMSE ratio; 93.3% seed wins</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Broad benchmarks</td><td>Unreleased execution details</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 7/12. Six tests pass.</p></div>
````
