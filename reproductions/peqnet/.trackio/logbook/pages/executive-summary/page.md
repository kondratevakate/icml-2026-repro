# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e9a5c5a86ca5", "created_at": "2026-07-29T15:43:53+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T15:43:54+00:00"}
-->
**Outcome.** Six claims were audited against arXiv
`2605.14284v2` and independent CPU checks. C4 is **VERIFIED with
qualification**: the MMD-to-MDS policy representation works and preserves
distance ordering, but does not provide the exact identity used later in the
proof. C5 is **FALSIFIED as a uniform Lipschitz guarantee** under the stated
assumptions. C1-C3 and C6 are **INCONCLUSIVE** because code and exact protected
data pipelines are unavailable.

## Scope and cost

| Item | Value |
| --- | --- |
| GPU | None |
| Protected medical data | None used or published |
| Independent policies | 5 |
| Synthetic histories | 240 |
| Unit tests | 4 passing |
| Prepared score | 4/12 |
| Conservative expected score | 4/12 |

The paper is medically relevant: its motivating and real-data setting is
longitudinal treatment-policy evaluation in MIMIC-III/MIMIC-IV.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_49d776ee1f4d", "created_at": "2026-07-29T15:43:54+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T15:43:54+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #cfd7df;padding:20px;background:#fff;color:#17202a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">PEQ-Net: independent embedding and theorem audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | bIcz7bIZSo | arXiv 2605.14284v2 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claims</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1-C3 empirical RMSE</td><td>No author code; protected input; printed DGP singular at i=1</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">C4 policy embedding</td><td>5 policies; independent MMD + metric MDS; rank correlation 1.0</td><td style="color:#137a46;font-weight:700">VERIFIED*</td></tr>
<tr><td style="padding:6px">C5 remainder theorem</td><td>Pair-specific constant, MDS mismatch, bounded-targeting counterexample</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">C6 MIMIC-IV case</td><td>No cohort SQL/item map; complete restricted data pending</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table>
<p style="font-size:12px;margin:12px 0 0">*Pipeline verified; exact MDS=MMD identity is not. Prepared 4/12.</p>
</div>
````
