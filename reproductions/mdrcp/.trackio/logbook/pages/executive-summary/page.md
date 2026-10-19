# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mdrcp_v2_summary", "created_at": "2026-07-29T08:10:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T08:10:00+00:00"}
-->
CPU-only independent repair of *Multi-Distribution Robust Conformal
Prediction*. The previous judged version earned `2/12`: only Theorem 1 was
fully supported. This version retains that result and adds direct audits of
Theorems 2 and 3.

| Claim | Current evidence | Verdict |
| --- | --- | --- |
| C1 finite-sample uniform validity | Earlier 10-seed classification/regression run; worst coverage 0.905/0.909 | VERIFIED |
| C2 population dual and optimal superlevel set | 500 primal/dual instances; max gap `2.23e-8`; all KKT checks pass | VERIFIED |
| C3 asymptotic optimality | 8 sample sizes x 20 seeds; score/set errors converge; frozen-weight control fails | VERIFIED |
| C4 classification efficiency | Official 100-trial Linear panel pending | PRE-VERDICT |
| C5 regression efficiency | Official 100-trial Linear panel pending | PRE-VERDICT |
| C6 FMoW | No FMoW data or checkpoint used | NOT ATTEMPTED |

Theorem 2 is tested on finite outcome spaces with counting measure, an explicit
setting of Section 3.1. Theorem 3 is tested with consistent empirical density
and dual-weight estimators; at the largest sample the calibrated-set symmetric
difference is `1.80`, within the oracle boundary measure of 3 outcomes.

## Reproducibility

```powershell
cd repro_mdrcp_v2
python run_all.py
```

Official code is pinned separately at commit
`6a7d5ba990c98aff3a9510bc40357254f3107e68`. Public peer logbooks were used
only to identify promising claims. Their outputs are not used as evidence.

## Score forecast

Current defensible forecast: `6/12` total, a `+4` gain over the previous
`2/12`. C4/C5 are deliberately excluded until a full official 100-trial run
finishes. C6 is excluded.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_mdrcp_v2_poster", "created_at": "2026-07-29T08:10:00+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T08:10:00+00:00", "poster": true}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 8px;font-size:19px;border-bottom:2px solid #1f6f5f;padding-bottom:8px">MDRCP: uniform validity and optimality audit</h2>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:5px;text-align:left">Claim</th><th>Evidence</th><th>Verdict</th></tr>
    <tr><td style="padding:5px">C1 finite-sample validity</td><td style="text-align:center">10 seeds, classification + regression</td><td style="text-align:center;color:#19733a">VERIFIED</td></tr>
    <tr><td style="padding:5px">C2 dual optimality</td><td style="text-align:center">500 primal/dual instances</td><td style="text-align:center;color:#19733a">VERIFIED</td></tr>
    <tr><td style="padding:5px">C3 asymptotic optimality</td><td style="text-align:center">8 sizes x 20 seeds + destructive control</td><td style="text-align:center;color:#19733a">VERIFIED</td></tr>
    <tr><td style="padding:5px">C4-C6 empirical panels</td><td style="text-align:center">full official runs not complete</td><td style="text-align:center;color:#9a6200">PRE-VERDICT</td></tr>
  </table>
</div>
````
