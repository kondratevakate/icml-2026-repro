# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_759e7ede172e", "created_at": "2026-07-29T20:21:46+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T20:21:46+00:00"}
-->
**Outcome.** This CPU reproduction executes all three
diabetes environments, independently evaluates one released policy under
patient shift, audits the conditional safety theorem, and tests whether the
released predictive-shield contracts close.

| Claim | Result | Score |
| --- | --- | ---: |
| Unified simulator | Partial | 1/2 |
| Safety-generalization gap | Partial | 1/2 |
| BA-NODE accuracy | Not executed | 0/2 |
| Conditional theorem | Verified | 2/2 |
| Released shield | Partial | 1/2 |
| **Prepared** |  | **5/10** |

No leaderboard or peer reproduction was inspected or used as evidence.

Primary sources: [arXiv](https://arxiv.org/abs/2601.21094),
[OpenReview](https://openreview.net/forum?id=kSUGLBHd0T),
[GlucoSim](https://github.com/safe-autonomy-lab/GlucoSim), and
[GlucoAlg](https://github.com/safe-autonomy-lab/GlucoAlg).


---
<!-- trackio-cell
{"type": "figure", "id": "cell_a6d0b7fbb350", "created_at": "2026-07-29T20:21:46+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T20:21:46+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #cbd5e1;padding:20px;background:#fff;color:#17212b">
<h2 style="font-size:18px;margin:0 0 8px;border-bottom:3px solid #b42318;padding-bottom:8px">Diabetes safety generalization: independent CPU audit</h2>
<p style="font-size:12px">ICML 2026 | kSUGLBHd0T | arXiv 2601.21094 | Prepared 5/10</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef2f6"><th style="text-align:left;padding:6px">Evidence</th><th>Finding</th></tr>
<tr><td style="padding:6px">Simulator</td><td>22/22 tests; 3 deterministic environments</td></tr>
<tr><td style="padding:6px">OOD mechanism</td><td>TIR gap -11.00 pp; risk gap +1.99</td></tr>
<tr><td style="padding:6px">Theorem</td><td>320 boundary checks; conditional implication holds</td></tr>
<tr><td style="padding:6px">Release audit</td><td>Path, cohort-index, and soft-pruning divergences</td></tr>
</table></div>
````
