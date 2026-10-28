# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ab63bff20b65", "created_at": "2026-07-30T07:15:06+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T07:15:06+00:00"}
-->
**Outcome.** The official GLEAN release receives
**2/12 points**, equal to the
frozen forecast. Both upstream datasets are accessible, but the author
released no GLEAN code, trajectories, predictions, calibration rows, active
traces, or clinician annotations.

| Independent check | Result |
| --- | --- |
| MIMIC-IV-Ext payload checksums | 12/12 match |
| MIMIC-IV-Ext cases | 2400 |
| Discounted accumulation recurrence | 0.0e+00 difference |
| Printed Appendix A bound follows | False |
| Qwen3 diverticulitis prose/Table-1 values conflict | True |
| Prepared score | 2/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2603.02798),
[OpenReview](https://openreview.net/forum?id=FP23eFYhAy),
[guideline corpus](https://huggingface.co/datasets/epfl-llm/guidelines), and
[MIMIC-IV-Ext CDM](https://physionet.org/content/mimic-iv-ext-cdm/1.1/).
No leaderboard or third-party verdict contributes evidence. Credentialed
clinical rows are excluded from this Space.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_499c4092da29", "created_at": "2026-07-30T07:15:06+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T07:15:06+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #c8d2dc;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #2563a6;padding-bottom:8px">GLEAN: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Source datasets</td><td>Guidelines metadata and credentialed 2,400-case CDM release verified</td></tr><tr><td>Method</td><td>Accumulation equation executable; Appendix bound has an epsilon-squared error</td></tr><tr><td>Main table</td><td>Not reproducible; prose conflicts with Table 1 in all five quoted example values</td></tr><tr><td>Outputs/study</td><td>No trajectories, predictions, active traces, Best-of-N groups, or clinician ratings</td></tr></table><p><b>Prepared 2/12</b>. No peer verdicts used.</p></div>
````
