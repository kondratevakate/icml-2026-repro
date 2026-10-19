# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9ebc0c10c8e1", "created_at": "2026-07-30T06:50:26+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T06:50:26+00:00"}
-->
**Outcome.** The official GlobalHealthAtlas release
receives **6/12 points**, equal
to the frozen forecast. The cached benchmark is unusually auditable: weighted
aggregation of 16 detailed model exports reconstructs all corresponding main
table values. The public corpus, expert gold annotations, and repeated
evaluator runs are not among the linked artifacts.

| Item | Independent result |
| --- | --- |
| GitHub revision | `23edda8517ed95e3a3db4fda0fc0fc53546532cb` |
| Released result CSVs | 26 |
| Main-table models reconstructed | 16/16 |
| Transfer cells matching paper rounding | 11/12 |
| Evaluator adapter SHA matches Hub | True |
| Public-Model adapter SHA matches Hub | True |
| Prepared score | 6/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2602.00491),
[OpenReview](https://openreview.net/forum?id=02Nq73lCe6),
[official repository](https://github.com/Jan8217/GlobalHealthAtlas),
[Public-Evaluator](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Evaluator),
and [Public-Model](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Model).
No leaderboard or third-party verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_aff66ced559d", "created_at": "2026-07-30T06:50:26+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T06:50:26+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #c8d2dc;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #2563a6;padding-bottom:8px">GlobalHealthAtlas: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Corpus</td><td>Not released under the linked official artifacts</td></tr><tr><td>Main benchmark</td><td>16/16 detailed exports reconstruct paper rounding</td></tr><tr><td>Transfer</td><td>11/12 cells match; Qwen-8B GPQA discrepancy</td></tr><tr><td>Adapters</td><td>Both safetensors payloads match Hub LFS SHA-256</td></tr></table><p><b>Prepared 6/12</b>. No peer verdicts used.</p></div>
````
