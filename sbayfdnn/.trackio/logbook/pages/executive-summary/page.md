# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_baa00c400959", "created_at": "2026-07-30T07:12:50+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T07:12:50+00:00"}
-->
**Outcome.** This independent CPU/source audit prepares
**4/12**. The released
spike-and-slab selection mechanism is verified. Theorem 5.4 is falsified as
stated by its unconstrained parameter bound. Theorems 5.7 and 5.9 have proof
gaps but remain inconclusive, and the unreleased ECG/Tecator pipelines remain
unexecuted.

| Item | Value |
| --- | --- |
| Prepared score | 4/12 |
| GPU | None |
| Medical data | None |
| Independent tests | 5 passing |
| Author commit | `276d87949c8b973bf8b6748aa8df4f56086e063e` |

No leaderboard or peer reproduction was inspected or used.

Primary sources: [arXiv](https://arxiv.org/abs/2602.20651),
[OpenReview](https://openreview.net/forum?id=3IFIedDIoN), and
[official code](https://github.com/mengyunwu2020/sBayFDNN).


---
<!-- trackio-cell
{"type": "figure", "id": "cell_59cbc0321f60", "created_at": "2026-07-30T07:12:50+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T07:12:50+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #c9d2d8;padding:20px;background:#fff;color:#17212b">
<h2 style="font-size:18px;margin:0 0 8px;border-bottom:3px solid #176b5b;padding-bottom:8px">sBayFDNN: independent method and theorem audit</h2>
<p style="font-size:12px">ICML 2026 | 3IFIedDIoN | arXiv 2602.20651 | Prepared 4/12</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#edf3f1"><th style="text-align:left;padding:6px">Evidence</th><th>Finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">PIP and regions</td><td>1,201 Bayes checks; exact interval mapping</td><td style="color:#176b5b;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Theorem 5.4</td><td>Unconstrained E_n admits shrinking-network counterexample</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Theorem 5.7</td><td>Statement inequality is opposite proof requirement</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">Theorem 5.9</td><td>Unstated posterior-to-MAP premise</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">ECG / Tecator</td><td>Release pipelines absent</td><td>NOT EXECUTED</td></tr>
</table></div>
````
