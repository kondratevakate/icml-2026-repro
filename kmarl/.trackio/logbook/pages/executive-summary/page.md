# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_aa274b9cf1c0", "created_at": "2026-07-20T08:30:50+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-20T08:30:50+00:00"}
-->
Independent from-the-paper reproduction, **numpy only, CPU, no GPU, fully synthetic (no data availability gate)**, of the paper's central claim: naive ARL/ADD estimators for changepoint detectors are biased under right-censoring, and the proposed Kaplan-Meier estimators (KM-ARL, KM-ADD) fix this. Detector: generalized Shiryaev-Roberts (GSR) with ground-truth statistics on a Gaussian stream where the log-likelihood ratio is exact (`x - 0.05`); ground truth ARL/ADD from effectively-infinite sequences per the paper's own method. **3 of 4 anchored claims cleanly confirmed across 10 seeds each; the 4th (A1) is honestly reported as INCONCLUSIVE**, exactly as the paper itself flags it as empirically hard to verify. **A3**: `|KM bias| < |LB bias|` in all 9 non-degenerate configurations tested, 10/10 seeds, at heaviest censoring KM relative bias -0.175 vs LB -0.522, and KM-ADD beats LB-ADD in 29/30 seed-runs. **A6**: in the regime the paper's own Figure 2 occupies, KM-ARL relative bias stays inside a pre-stated +/-5% band (-0.014 to -0.028) flat across changepoint prevalence 10-90%; outside that regime (an extrapolation zone the paper itself declares unbiased by no estimator) bias grows large, reported honestly rather than hidden. **A4**: tested on BOTH sides of the iff support condition -- satisfied gives bias converging to zero as N grows (50 to 20000: -0.023 to +0.003), violated gives bias plateauing at -0.107 and not vanishing. **A1**: NOT numerically established -- at small N estimation variance overshadows the bias (N=50: seed sd 2.65 vs mean bias -1.35), matching the paper's own caveat; decay toward zero is visible but the exponential rate is not. This is the same family as this project's cleanest prior result (censoring distorts naive evaluation metrics, scored 4/4 for survival model evaluation), here applied to changepoint detection. Process note: an initial WebFetch summary fabricated a plausible-sounding setup (invented CUSUM detector, threshold 370, changepoint at 2500); this was caught and discarded, and everything below comes from the actual PDF text extracted with PyMuPDF. Wall time ~2 min; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | GSR detector, ground-truth statistics, Gaussian stream, N up to 20000, 10 seeds per cell, Theorems 4.1/4.3/4.4 and Eq. 11 (claims A1/A3/A4/A6) | Full Fig. 2-4 sweep incl. Poisson process model, additional detectors (CUSUM), real-world changepoint benchmarks |
| Hardware | CPU (numpy, vectorised GSR recursion) | CPU (method is not GPU-bound) |
| Compute time | ~2 min | tens of minutes across full sweep |
| Cost | ~$0 | ~$0 |
| Outcome | A3 confirmed (9/9 configs, 10/10 seeds); A6 confirmed in-scope (pre-stated +/-5% band); A4 confirmed both directions of the iff condition; A1 inconclusive (paper's own caveat) | - |


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
    <tr><td style="padding:4px">A1: exponential decay rate of bias</td><td style="padding:4px;text-align:center">decay visible, rate not established</td><td style="padding:4px;text-align:center;color:#e67e22">INCONCLUSIVE</td></tr>
  </table>
  <p style="font-size:12px;color:#555;margin-top:8px">Same family as the survival-model-evaluation censoring-bias result (4/4): naive metric estimation under censoring biases evaluation, here for changepoint detection ARL/ADD instead of survival metrics.</p>
</div>
````
