# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_cc1ade11d078", "created_at": "2026-07-29T16:17:42+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T16:17:42+00:00"}
-->
**Outcome.** Six claims from arXiv `2605.01452v1` were
audited without leaderboard or peer-reproduction evidence. C1 is
**VERIFIED**. C4 is **FALSIFIED AS WRITTEN** by an exact exchangeable-score
counterexample. C2 and C3 are **NOT ESTABLISHED AS WRITTEN** because their
proofs require a corrected random-error term or unstated regularity. C5 and
C6 remain pending fresh canonical experiments.

## Scope and cost

| Item | Value |
| --- | --- |
| GPU | None |
| Protected medical data | None |
| Independent stability constructions | 4,000 |
| Rate-surrogate repetitions | 200,000 |
| Unit tests | 5 passing |
| Prepared score | 6/12 |
| Conservative expected score | 6/12 |

The paper is medically relevant through its DermaMNIST dermatoscopy and
TissueMNIST kidney-cortex microscopy experiments.

Provenance: [OpenReview](https://openreview.net/forum?id=lSMTccAN61),
[arXiv](https://arxiv.org/abs/2605.01452), and the pinned
[author repository](https://github.com/OswinMin/SLCP) at commit
`84118ddf19efe211250a5f024c1be5a3c8f47b4b`.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_9da5524b176b", "created_at": "2026-07-29T16:17:42+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T16:17:42+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #cfd7df;padding:20px;background:#fff;color:#17202a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">StCP: independent theorem and stability audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | lSMTccAN61 | arXiv 2605.01452v1 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 set stability</td><td>Total-variance audit, 4,000 constructions</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">C2 marginal bound</td><td>Random-delta proof counterexample</td><td style="color:#7a5c13;font-weight:700">NOT ESTABLISHED</td></tr>
<tr><td style="padding:6px">C3 variance rate</td><td>Rate sanity passes; assumptions and moment gap remain</td><td style="color:#7a5c13;font-weight:700">NOT ESTABLISHED</td></tr>
<tr><td style="padding:6px">C4 selected lambda</td><td>27/31 &lt; 0.88 under continuous exchangeability</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">C5-C6 experiments</td><td>Fresh canonical runs pending</td><td style="color:#7a5c13;font-weight:700">PENDING</td></tr>
</table>
<p style="font-size:12px;margin:12px 0 0">Prepared 6/12. No peer reproductions used.</p>
</div>
````
