# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1d7e842756f2", "created_at": "2026-07-29T16:40:28+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T16:40:28+00:00"}
-->
**Outcome.** The pinned author implementation was audited
against six paper claims without using leaderboard or peer-reproduction
evidence. Five architecture claims are **FALSIFIED IN THE RELEASED
IMPLEMENTATION** by executable shape, mutation, invariance, and gradient
tests. The five-dataset empirical claim remains inconclusive.

| Item | Value |
| --- | --- |
| Prepared score | 10/12 |
| GPU | None |
| Medical data | None |
| Unit tests | 6 passing |
| Author commit | `418da50664338bc1d766394ee9c231496ab4de97` |

The identity Mamba substitute is used only to expose the released integration
code's tensor axes and dependencies. No accuracy conclusion depends on it.

Provenance: [arXiv](https://arxiv.org/abs/2605.24961) and
[author repository](https://github.com/zhangda1018/MedMamba).


---
<!-- trackio-cell
{"type": "figure", "id": "cell_76ccc7a20823", "created_at": "2026-07-29T16:40:28+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T16:40:28+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">MedMamba: released implementation audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | qPqJH0heR0 | arXiv 2605.24961v1 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Executable finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">MCE</td><td>Channel axis collapsed; groups=1</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Difference</td><td>First step equals x[0], not zero</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Frequency filter</td><td>D weights broadcast over all FFT bins</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Adaptive graph</td><td>Input invariant; no output or classification gradient path</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Spatial SGM</td><td>Scans time; adjacency unused</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Five datasets</td><td>Fresh runs unavailable</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 10/12. No peer reproductions used.</p></div>
````
