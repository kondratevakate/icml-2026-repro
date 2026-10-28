# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_485530d2daae", "created_at": "2026-07-30T08:21:24+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T08:21:24+00:00"}
-->
**Outcome.** The release receives **4/10 points**, one point
below the frozen 5/10 forecast. The official checkpoint contains the coherent
five-way task partition and aggregate replay-free EWC state, but it uses CRP
alpha 2.0 rather than the reported 5.0 and provides no metric outputs.

| Independent check | Result |
| --- | --- |
| Tasks / discovered modalities | 16 / 5 |
| Checkpoint alpha / paper alpha | 2.0 / 5.0 |
| Fixed-Gaussian tail error | 0.0327 |
| Released run-level metric files | 0 |
| Deterministic audit tests | 9 passed |
| Prepared score | 4/10 |

Primary sources: [arXiv](https://arxiv.org/abs/2605.20297),
[official code](https://github.com/zygao930/MedCRP-CL), and
[official checkpoint](https://huggingface.co/clg-g/MedCRP-CL). No leaderboard
or third-party verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_5199fd76e58a", "created_at": "2026-07-30T08:21:24+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T08:21:24+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #b9c6cc;padding:20px;background:#fff;color:#17232a"><h2 style="border-bottom:3px solid #0f766e;padding-bottom:8px">MedCRP-CL: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Partition</td><td>Official state: 16 tasks, 5 coherent groups; alpha mismatch 2.0 vs 5.0</td></tr><tr><td>Mechanics</td><td>Modality LoRA isolation and aggregate EWC state independently checked</td></tr><tr><td>Theory</td><td>Fixed Gaussian overlap contradicts the appendix zero-error step</td></tr><tr><td>Empirics</td><td>No processed splits, metric outputs, predictions, or baselines released</td></tr></table><p><b>Prepared 4/10</b>. No peer verdicts used.</p></div>
````
