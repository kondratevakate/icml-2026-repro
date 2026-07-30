# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c8e5cbb26659", "created_at": "2026-07-30T08:35:09+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T08:35:09+00:00"}
-->
**Outcome.** The release receives **5/12 points**, one point
above the frozen 4/12 forecast. The self-reference and containment mechanisms
execute on deterministic synthetic pathology inputs, but canonical POT crashes,
the minimally completed transport plan is N-scaled, and the empirical tables
have no released run-level artifacts.

| Independent check | Result |
| --- | --- |
| Self-reference precision / recall | 1.000 / 0.607 |
| Canonical POT | `AttributeError: 'OptimalTransport' object has no attribute 'numItermax'` |
| Runtime-completed real / target mass | 2.400 / 0.600 |
| Empty refinement input | `IndexError: list index out of range` |
| Released empirical artifacts | 0 |
| Deterministic audit tests | 8 passed |
| Prepared score | 5/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2511.19953),
[official code](https://github.com/Y-Research-SBU/SPROUT), and the
[project page](https://y-research-sbu.github.io/SPROUT/). No leaderboard or
third-party verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_649017857ca3", "created_at": "2026-07-30T08:35:09+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T08:35:09+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #d2c3b3;padding:20px;background:#fffaf5;color:#29211b"><h2 style="border-bottom:3px solid #9f4f28;padding-bottom:8px">SPROUT: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Layer</th><th>Result</th></tr><tr><td>Self-reference</td><td>Synthetic H&amp;E precision 1.000, recall 0.607</td></tr><tr><td>Partial OT</td><td>Canonical crash; runtime-completed plan is N-scaled</td></tr><tr><td>Refinement</td><td>Containment penalty works; empty input crashes</td></tr><tr><td>Empirics</td><td>No predictions, processed splits, metrics, or repeat outputs</td></tr></table><p><b>Prepared 5/12</b>. No peer verdicts used.</p></div>
````
