# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6d88fd576cd6", "created_at": "2026-07-30T07:41:34+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-30T07:41:34+00:00"}
-->
**Outcome.** The CAML release receives
**4/12 points**, below the
frozen forecast of 7/12. The aligned-offset equation and delay schedule match
the official implementation, but the headline theorem is false as stated and
five released benchmark settings differ from the paper.

| Independent check | Result |
| --- | --- |
| Non-unique operator has connected solution set | False |
| Closed-form offset absolute difference | 0.0e+00 |
| Degenerate zero-coefficient offset finite | False |
| Paper/code benchmark-setting mismatches | 5 |
| Mini Heat runtime | 2.26 s for 10 epochs |
| Prepared score | 4/12 |

Primary sources: [arXiv](https://arxiv.org/abs/2605.25001) and
[official code](https://github.com/YichenLuo-0/CAML). No leaderboard or
third-party verdict contributes evidence.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_8e0f935c3ffb", "created_at": "2026-07-30T07:41:34+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-30T07:41:34+00:00"}
-->
````html
<div style="font-family:system-ui,Arial;max-width:960px;border:1px solid #bbc7d1;padding:20px;background:#fff;color:#18222d"><h2 style="border-bottom:3px solid #7c3aed;padding-bottom:8px">CAML: independent release audit</h2><table style="font-size:12px;width:100%"><tr><th>Evidence layer</th><th>Result</th></tr><tr><td>Theorem</td><td>False as stated: N(u)=u², f=1 has two isolated disconnected solutions</td></tr><tr><td>Method</td><td>Offset and delay schedule exactly conform; zero-denominator case returns NaN</td></tr><tr><td>Empirics</td><td>Five paper/code settings differ; no cached outputs or full baseline suite</td></tr><tr><td>Runtime</td><td>Official Heat path executes after adding undocumented overrides dependency</td></tr></table><p><b>Prepared 4/12</b>. No peer verdicts used.</p></div>
````
