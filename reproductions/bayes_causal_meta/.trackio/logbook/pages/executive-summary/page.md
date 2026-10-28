# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2e00f388663c", "created_at": "2026-07-30T08:08:15+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T08:08:15+00:00"}
-->
**Outcome.** The release receives **5/10 points**, matching
the frozen forecast. Proposition 4 is sound for the paper's linear Gaussian
prior, but the released normalized prior is discontinuous at zero and
Appendix Theorem 5 drops the condition number of `W`.

| Independent check | Result |
| --- | --- |
| Released prior jump at zero | 1.0 |
| Paper Gaussian-TV bounds hold | True |
| Theorem-constant counterexample | True |
| Same-seed validation AUROC range | 0.0745 |
| UK Biobank data released | False |
| Prepared score | 5/10 |

Primary sources: [arXiv](https://arxiv.org/abs/2602.19788) and
[official code](https://github.com/lottamakinen/causal-meta-learning). No
leaderboard or third-party verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_59b2e07d89c3", "created_at": "2026-07-30T08:08:15+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T08:08:15+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #bbc7d1;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #2563eb;padding-bottom:8px">Bayesian causal meta-learning: independent audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Prior</td><td>Release normalizes every nonzero Wz, creating scale invariance and a jump at zero</td></tr><tr><td>Proposition 4</td><td>Gaussian-TV and error-decomposition bounds verified for the paper model</td></tr><tr><td>Theorem 5</td><td>Appendix constant omits cond(W); explicit 2D counterexample</td></tr><tr><td>Empirics</td><td>Toy training and BALD smoke work; UK Biobank artifacts absent</td></tr></table><p><b>Prepared 5/10</b>. No peer verdicts used.</p></div>
````
