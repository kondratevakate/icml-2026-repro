# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a92adfbcf11a", "created_at": "2026-07-19T08:18:56+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-19T08:18:56+00:00"}
-->
Independent from-the-paper reproduction on the paper's own synthetic model, **numpy + torch, CPU, no GPU**. **Both reproduced claims hold.** Claim 1: with *unaligned* groups (the realistic case where you do not know which covariate drives miscoverage) CovGap is effectively blind — it reports 0.005 against a true L1 miscoverage of 0.095 — while L1-ERT with a learned classifier recovers 0.087, i.e. **92% of the truth vs 5%**. Claim 2: the over/under decomposition matches theory closely (over 0.0445 vs 0.0460; under 0.0464 vs 0.0488). Two bonus checks: ERT's lower-bound property is visible as monotone convergence from below (0.021 -> 0.087), and on oracle conditionally-valid sets ERT correctly returns ~0. Wall time ~5 min; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Both claims on the paper's synthetic model (X~U[-1,1]^8, Y~N(0,sigma(X1))), exact analytic ground truth | Also the real-data benchmark suite and the full classifier zoo / released ERT package |
| Hardware | CPU (numpy + torch MLP) | CPU (paper's setting is also light) |
| Compute time | ~5 min | hours across datasets/classifiers |
| Cost | ~$0 | ~$0 |
| Outcome | Claims 1-2 reproduced; the precise boundary of claim 1 documented (oracle groups make CovGap accurate) | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_4c331875f8bd", "created_at": "2026-07-19T08:18:56+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-19T08:18:56+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #1f6f8b;padding-bottom:8px">Conditional Coverage Diagnostics: ERT finds violations that group metrics miss</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview vaApZm6MKM &middot; arXiv 2512.11779 &middot; reproduced by kondratevakate &middot; CPU</div>
  <p style="font-size:13px">Synthetic model with <b>exact</b> conditional coverage: true L1 miscoverage = <b>0.0948</b>.</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">n test</th><th style="padding:4px">L1-ERT (learned)</th><th style="padding:4px">CovGap, oracle groups</th><th style="padding:4px">CovGap, unaligned groups</th></tr>
    <tr><td style="padding:4px">1 000</td><td style="padding:4px;text-align:center">0.021</td><td style="padding:4px;text-align:center">0.095</td><td style="padding:4px;text-align:center">0.023</td></tr>
    <tr><td style="padding:4px">10 000</td><td style="padding:4px;text-align:center">0.069</td><td style="padding:4px;text-align:center">0.096</td><td style="padding:4px;text-align:center">0.008</td></tr>
    <tr><td style="padding:4px">30 000</td><td style="padding:4px;text-align:center;color:#27ae60"><b>0.087</b></td><td style="padding:4px;text-align:center">0.095</td><td style="padding:4px;text-align:center;color:#c0392b"><b>0.005</b></td></tr>
  </table>
  <p style="font-size:13px;margin-top:8px">Without being told where to look, ERT recovers <b>92%</b> of the true violation; a group metric with unaligned groups sees <b>5%</b>. Over/under split: 0.0445 / 0.0464 (theory 0.0460 / 0.0488).</p>
  <p style="font-size:12px;color:#555">Why it matters clinically: marginal coverage can look fine while a patient subgroup is badly under-covered. ERT surfaces that without pre-specifying the subgroup.</p>
</div>
````
