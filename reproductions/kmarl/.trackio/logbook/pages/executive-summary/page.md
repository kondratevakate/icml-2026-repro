# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_aa274b9cf1c0", "created_at": "2026-07-20T08:30:50+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-20T08:30:50+00:00"}
-->
Independent from-the-paper reproduction, **numpy only, CPU, fully synthetic**, of KM-ARL/KM-ADD under right censoring. **Five of six anchored claims now have direct evidence.** Theorems 4.1 and 4.2 pass independent derivation checks plus 500,000-dataset-per-cell finite-sample audits over `N=2..64`; removing independent censoring makes the claimed upper bound fail by more than 10x. Theorems 4.3/4.4 retain their 10-seed truncation-bias verification, and Eq. 11 is tested on both sides of its support condition. The simulation claim now includes both process families: the Gaussian sweep covers 10%-90% changepoint prevalence, while the Poisson Figure 13 repair uses all 14 official GSR thresholds, `N=10,000`, five seeds, and both fixed and irregular lengths. In the censoring-active non-extrapolation zone, KM beats LB on normalized ARL-ADD curve error in **95.0%** of seed-cells with median distance ratio **0.567**; in the low-censoring null zone the ratio is **1.001**. The WISDM variance claim remains explicitly not attempted.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Theorems 4.1-4.4, Eq. 11, Gaussian Figures 2-4, and Poisson Figure 13 | Additional detectors (CUSUM) and WISDM real-data variance claim |
| Hardware | CPU (numpy, vectorised GSR recursion) | CPU (method is not GPU-bound) |
| Compute time | ~2 min | tens of minutes across full sweep |
| Cost | ~$0 | ~$0 |
| Outcome | 5/6 anchored claims supported; WISDM not attempted | - |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_ab4ef47e12f0", "created_at": "2026-07-20T08:30:50+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-20T08:30:50+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:18px;border-bottom:2px solid #117864;padding-bottom:8px">KM-ARL / KM-ADD: fixing censoring bias in changepoint-detector evaluation</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview LhGxRnGmGJ &middot; arXiv 2605.18798 &middot; PMLR 306 &middot; reproduced by kondratevakate &middot; CPU, synthetic, analytic ground truth</div>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">claim</th><th style="padding:4px">result</th><th style="padding:4px">verdict</th></tr>
    <tr><td style="padding:4px">A3: KM truncation bias &lt; conventional</td><td style="padding:4px;text-align:center">9/9 configs, 10/10 seeds; -0.175 vs -0.522</td><td style="padding:4px;text-align:center;color:#27ae60">CONFIRMED</td></tr>
    <tr><td style="padding:4px">A6: accurate across 10-90% censoring</td><td style="padding:4px;text-align:center">rel. bias -0.014..-0.028, flat, within +/-5%</td><td style="padding:4px;text-align:center;color:#27ae60">CONFIRMED (in scope)</td></tr>
    <tr><td style="padding:4px">A4: asymptotically unbiased iff support&lt;boundary</td><td style="padding:4px;text-align:center">N=20000: -0.003 (sat.) vs -0.107 (viol.)</td><td style="padding:4px;text-align:center;color:#27ae60">CONFIRMED both sides</td></tr>
    <tr><td style="padding:4px">A1/A2: finite-sample bias bounds</td><td style="padding:4px;text-align:center">12/12 bound cells pass; destructive control fails</td><td style="padding:4px;text-align:center;color:#27ae60">VERIFIED</td></tr>
    <tr><td style="padding:4px">A6 Poisson completion</td><td style="padding:4px;text-align:center">95.0% stress wins; median ratio 0.567</td><td style="padding:4px;text-align:center;color:#27ae60">VERIFIED</td></tr>
  </table>
  <p style="font-size:12px;color:#555;margin-top:8px">Same family as the survival-model-evaluation censoring-bias result (4/4): naive metric estimation under censoring biases evaluation, here for changepoint detection ARL/ADD instead of survival metrics.</p>
</div>
````
