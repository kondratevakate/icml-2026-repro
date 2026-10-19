# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3674e1535249", "created_at": "2026-07-30T07:55:08+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T07:55:08+00:00"}
-->
**Outcome.** The DC-PnPDP release receives **5/12 points**,
below the frozen forecast of 6/12. The effective-weight dual fixed-point
theorem is algebraically sound, but the released CT example removes the ADMM
penalty and the repository contains no MRI implementation or cached metrics.

| Independent check | Result |
| --- | --- |
| Dual objective residual | 2.22e-16 |
| Loose original-objective residual | 0.15225 |
| Zero-penalty anchor difference | 0.0e+00 |
| SH PSD relative RMSE | 0.0931 |
| MRI implementation released | False |
| Prepared score | 5/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2602.23214) and
[official code](https://github.com/duchenhe/DC-PnPDP). No leaderboard or
third-party verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_8fbd294e96d8", "created_at": "2026-07-30T07:55:08+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T07:55:08+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #b8c4cc;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #0f766e;padding-bottom:8px">DC-PnPDP: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Theory</td><td>Dual theorem verifies with lambda=rho gamma; loose theorem has scaling and interpretation gaps</td></tr><tr><td>Release</td><td>Example w_tik=0 versus paper LACT value 1e-5; no MRI path</td></tr><tr><td>SH</td><td>Finite mechanism, but paper/code sampling differs and native-order smoothing breaks FFT symmetry</td></tr><tr><td>Empirics</td><td>No data, reconstructions, metric outputs, ablation outputs, or timing artifacts</td></tr></table><p><b>Prepared 5/12</b>. No peer verdicts used.</p></div>
````
