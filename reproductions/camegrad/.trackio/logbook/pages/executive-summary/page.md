# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6c76b929ea34", "created_at": "2026-07-29T18:36:44+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T18:36:44+00:00"}
-->
**Outcome.** This CPU-only audit evaluates five claims
against the paper equations and pinned public artifact. The exact two-task
interaction identity is verified. The unconditional Stage 1 Pareto guarantee
is falsified by an exact counterexample. Post-fusion validity receives a scope
qualification, Stage 2 is partially verified, and the clinical claim is
inconclusive.

| Item | Value |
| --- | --- |
| Prepared score | 6/10 |
| GPU | None |
| Medical data | None |
| Unit tests | 6 passing |
| Author commit | `79059e39060d13ef6b6cb2ea9ad7a5d8e2519c83` |

The conclusions are limited to the stated equations and released artifact.
No leaderboard or peer reproduction is used as evidence.

Provenance: [arXiv](https://arxiv.org/abs/2605.22635) and
[author repository](https://github.com/vpsg-research/CAME-Grad).


---
<!-- trackio-cell
{"type": "figure", "id": "cell_9927297977b6", "created_at": "2026-07-29T18:36:44+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T18:36:44+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">CAME-Grad: mathematical guarantee audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | T7y2wavrFM | arXiv 2605.22635v2 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Executable finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">Interaction identity</td><td>1,000 trials; opposing gradients cancel</td><td style="color:#067647;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Stage 1 guarantee</td><td>Exact optimum harms one incompatible task</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Stage 2 scaling</td><td>Covariance holds; epsilon misses exact norm</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Stage 3 validity</td><td>Legal fusion reintroduces task conflict</td><td style="color:#7a5c13;font-weight:700">QUALIFIED</td></tr>
<tr><td style="padding:6px">Clinical efficacy</td><td>Core optimizer and weights withheld</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 6/10. No peer reproductions used.</p></div>
````
