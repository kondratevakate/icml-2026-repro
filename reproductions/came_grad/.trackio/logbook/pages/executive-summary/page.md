# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b03cd395cca4", "created_at": "2026-07-29T17:12:57+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T17:12:57+00:00"}
-->
**Outcome.** CAME-Grad's printed equations pass an independent
mechanics audit, preparing **2/10 points**, exactly as forecast. The SDE causal
claims are not established by those equations, and all clinical claims remain
inconclusive because the official release omits the optimizer, dataset module,
weights, requirements, and runnable evaluation bundle.

| Item | Result |
| --- | --- |
| Random equation checks | 500 |
| Maximum trust-region residual | `4.44e-16` |
| Fusion residual | `0.0` |
| Unit/mutation tests | 5 passing |
| GPU / medical data | none |
| Prepared score | 2/10 |

Primary sources: [arXiv](https://arxiv.org/abs/2605.22635),
[OpenReview](https://openreview.net/forum?id=T7y2wavrFM), and the
[official repository](https://github.com/vpsg-research/CAME-Grad). No
leaderboard or third-party reproduction verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_1b179508069a", "created_at": "2026-07-29T17:12:57+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T17:12:57+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;border-bottom:3px solid #a13b2a;padding-bottom:8px">CAME-Grad: independent equation and release audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | T7y2wavrFM | arXiv 2605.22635v2 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#f8efed"><th style="padding:6px;text-align:left">Claim</th><th>Evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 mechanics</td><td>500 equation checks; trust/norm/fusion invariants</td><td style="color:#137a46;font-weight:700">VERIFIED WITH QUALIFICATIONS</td></tr>
<tr><td style="padding:6px">C2 SDE mechanism</td><td>Equal norms produce covariance trace 0 vs 0.5</td><td style="color:#b36b00;font-weight:700">NOT ESTABLISHED</td></tr>
<tr><td style="padding:6px">C3-C5 experiments</td><td>Core optimizer and runnable artifacts withheld</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 2/10. Forecast calibrated. No peer verdicts used.</p></div>
````
