# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_351b33e0e312", "created_at": "2026-07-18T17:16:56+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-18T17:16:56+00:00"}
-->
Independent from-the-paper reproduction of this solvable spiked-cumulant model, on synthetic data (the paper's own setting), **numpy, CPU only, no GPU**. **Claim 2 (test loss misaligns) reproduces convincingly**; **Claim 1 (activation-selective recovery) reproduces cleanly at k*=2 and directionally-but-marginally at k*=3** — matching the paper's own near-floor k*=3 regime (Fig. 7). Bonus: PCA == linear autoencoder (Prop. 2.1); quadratic fails even the visible spike (Table 1). Paper-scale sweep ~40 min on CPU; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Both claims on the synthetic spiked model; d=800, alpha in {15,30}, 800 epochs, k*=2 and k*=3 | Paper scale d=2000, alpha to ~30, 1200 epochs, AMP + gradient-flow theory, all 8 activations |
| Hardware | CPU (numpy, analytic gradient + hand-rolled Adam) | CPU/GPU at larger d |
| Compute time | ~40 min | hours |
| Cost | ~$0 | ~$0 |
| Outcome | Claim 2 reproduced; Claim 1 clean at k*=2, marginal at k*=3 (as in the paper) | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_55179c7ce8f3", "created_at": "2026-07-18T17:16:56+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-18T17:16:56+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #2c3e50;padding-bottom:8px">Solvable model: nonlinear AEs see what PCA cannot; test loss misaligns</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview wm3ABfhE7P &middot; arXiv 2602.10680 &middot; reproduced by kondratevakate &middot; CPU only</div>
  <p style="font-size:13px"><b>Claim 1</b> — activation-selective recovery of hidden spike v* (theta_v vs d^-1/2 floor=0.035):</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">activation</th><th style="padding:4px">k*=2</th><th style="padding:4px">k*=3</th></tr>
    <tr><td style="padding:4px">ReLU</td><td style="padding:4px;text-align:center;color:#27ae60"><b>0.184 recovers</b></td><td style="padding:4px;text-align:center;color:#c0392b">0.011 floor</td></tr>
    <tr><td style="padding:4px">tanh</td><td style="padding:4px;text-align:center;color:#c0392b">0.028 floor</td><td style="padding:4px;text-align:center;color:#e67e22">0.058 marginal</td></tr>
    <tr><td style="padding:4px">linear / PCA</td><td style="padding:4px;text-align:center;color:#c0392b">floor</td><td style="padding:4px;text-align:center;color:#c0392b">floor</td></tr>
  </table>
  <p style="font-size:13px;margin-top:8px"><b>Claim 2</b> — linear AE has lower test reconstruction loss (799.1 &lt; 800.1) yet worse downstream error (0.497 vs 0.451): the metric misaligns with representation quality.</p>
</div>
````
