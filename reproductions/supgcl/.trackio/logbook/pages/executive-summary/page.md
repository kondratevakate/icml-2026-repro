# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5d1436b01968", "created_at": "2026-07-30T07:21:19+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T07:21:19+00:00"}
-->
**Outcome.** This independent CPU/source audit prepares
**4/12**. Theorem 1's KL
decomposition and Corollary 1's unsupervised-GCL limit are verified. The
TCGA/LINCS empirical claims remain unexecuted.

| Item | Value |
| --- | --- |
| Prepared score | 4/12 |
| GPU | None |
| Private medical data | None |
| Independent tests | 5 passing |
| Author commit | `a384eeaeffaabfd140335e19a0633ab400ba3bcb` |

The full empirical path requires the official 4.3 GB graph archive, H100-class
pretraining, and repeated cross-validation. No leaderboard or peer
reproduction was inspected or used.

Primary sources: [arXiv](https://arxiv.org/abs/2505.17786),
[OpenReview](https://openreview.net/forum?id=0Y7itV1kb7),
[official code](https://github.com/shobioinfo/SupGCL), and
[official data](https://zenodo.org/records/15496012).


---
<!-- trackio-cell
{"type": "figure", "id": "cell_2dd4a8a87003", "created_at": "2026-07-30T07:21:19+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T07:21:19+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #c9d2d8;padding:20px;background:#fff;color:#17212b">
<h2 style="font-size:18px;margin:0 0 8px;border-bottom:3px solid #176b5b;padding-bottom:8px">SupGCL: independent theorem audit</h2>
<p style="font-size:12px">ICML 2026 | 0Y7itV1kb7 | arXiv 2505.17786 | Prepared 4/12</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#edf3f1"><th style="text-align:left;padding:6px">Evidence</th><th>Finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">Theorem 1</td><td>100 independent KL chain-rule probes</td><td style="color:#176b5b;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Corollary 1</td><td>25 temperature-limit probes plus analytic limit</td><td style="color:#176b5b;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Tables 2-3</td><td>4.3 GB data and long GPU runs required</td><td>NOT EXECUTED</td></tr>
<tr><td style="padding:6px">Embedding analysis</td><td>Pretrained checkpoints not released</td><td>NOT EXECUTED</td></tr>
</table></div>
````
