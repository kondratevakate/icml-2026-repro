# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bbd4bf751b95", "created_at": "2026-07-30T07:29:41+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T07:29:41+00:00"}
-->
**Outcome.** The official LVCG release receives
**2/12 points**, below the
frozen forecast of 4/12. The printed VCG geometry works independently and the
released record splits are disjoint, but the official Python package cannot
import because `lvcg/data` is absent.

| Independent check | Result |
| --- | --- |
| Official package import | `ModuleNotFoundError: No module named 'lvcg.data'` |
| Missing internal imports | 3 |
| Synthetic geometry check | True |
| Six task record splits disjoint | True |
| Pretrained checkpoints released | 0 |
| Prepared score | 2/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2605.31249),
[official code](https://github.com/BosonHwang/LVCG), and
[MIMIC-IV-ECG-Ext-ICD](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/).
No leaderboard or third-party verdict contributes evidence, and no clinical
waveforms are included.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_26ff8a07aa2f", "created_at": "2026-07-30T07:29:41+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T07:29:41+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #b9c6d3;padding:20px;background:#fff;color:#17212b"><h2 style="border-bottom:3px solid #b42318;padding-bottom:8px">LVCG: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Geometry</td><td>Independent ridge lift/project passes at 1e-4</td></tr><tr><td>Runtime</td><td>Official package cannot import: missing lvcg/data</td></tr><tr><td>Linear probing</td><td>Splits present; no checkpoint/results; PTB-XL Sub label order differs</td></tr><tr><td>Non-cardiac</td><td>Paper uses MIMIC-IV-ECG-Ext-ICD; code config uses AI-READI</td></tr></table><p><b>Prepared 2/12</b>. No peer verdicts used.</p></div>
````
