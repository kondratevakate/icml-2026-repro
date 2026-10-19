# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8c05cecb2fce", "created_at": "2026-07-30T08:55:39+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T08:55:39+00:00"}
-->
**Outcome.** The release receives **4/12 points**, one point
below the frozen 5/12 forecast. The outer folds and core evidential model pass,
but non-default lambda semantics, empty components, and the visualization
entrypoint fail; the empirical TCGA outputs are absent.

| Independent check | Result |
| --- | --- |
| Cohorts / outer folds / CSVs | 5 / 25 / 50 |
| Missing DSS endpoint rows | 86 |
| Synthetic gradient tensors finite | 12/12 |
| lambda=0.75 training/paper max gap | 0.500 |
| lambda=0.75 training survival maximum | 1.414 |
| Visualization entrypoint | `NameError: name 'create_embedding_model' is not defined` |
| Prepared score | 4/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2510.00053) and
[official code](https://github.com/YuchengXing99/DPsurv). No leaderboard or
third-party verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_b3eaec489584", "created_at": "2026-07-30T08:55:39+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T08:55:39+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #c9bfcf;padding:20px;background:#fcf9ff;color:#251d2a"><h2 style="border-bottom:3px solid #6d3f86;padding-bottom:8px">DPsurv: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Splits</td><td>25 folds; no case/slide leakage; 86 rows missing DSS endpoint</td></tr><tr><td>Mechanics</td><td>Synthetic dual-prototype forward, loss, and backward pass</td></tr><tr><td>Theory/code</td><td>Bel <= Pl; lambda paths disagree away from 0.5</td></tr><tr><td>Empirics</td><td>No features, checkpoints, predictions, summaries, or logs</td></tr></table><p><b>Prepared 4/12</b>. No peer verdicts used.</p></div>
````
