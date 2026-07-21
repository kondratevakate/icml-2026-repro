# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_71a1605fe1ae", "created_at": "2026-07-20T08:32:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-20T08:32:00+00:00"}
-->
Independent from-the-paper reproduction of MDCP (Multi-Distribution Robust Conformal Prediction) on the paper's own synthetic 3-source setting, **numpy + torch, CPU only, no GPU**. **Claim 1 (Theorem 1, finite-sample worst-case validity) is CONFIRMED**: MDCP achieves worst-case coverage 0.905 (classification) / 0.909 (regression) against nominal 0.90 over 10 seeds, matching the paper's own reported 90.25%, and this holds under every lambda tried (fitted or oracle). **Claim 2 (prediction-set size reduction, Fig. 2 and Fig. 5) is INCONCLUSIVE**: with a fitted covariate-dependent lambda we measure 4.73% (classification) / 8.71% (regression) reduction vs. the naive max-p baseline, far below the paper's claimed 34.39% / 22.44%; a follow-up oracle-lambda test (grid-searched, but restricted to the constant-lambda family) reaches 13.49% / 11.91% — nearly triple the fitted result in classification — which proves the fitted-lambda optimizer is a real, demonstrated bottleneck and leaves the ceiling for a faithful (covariate-dependent-spline) reproduction unmeasured. Verdict: not refuted, not confirmed. A side finding, independently useful: the paper's Eq. 7 p-value, implemented literally, yields the complement of the intended prediction set — likely a one-character typesetting error, not a substantive defect (see Claim 2 / Conclusion). Directly relevant to Kate's interest in worst-case/multi-group coverage guarantees, where "worst distribution" maps onto "worst scanner/site" in medical imaging. Paper-scale sweep (10 seeds x classification + regression + oracle) runs in minutes on CPU; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Both settings (linear classification C=6, linear regression), K=3 sources, d=10, alpha=0.1, 10 seeds; Theorem 1 (worst-case validity) and the size-reduction claim (Fig. 2/Fig. 5), plus an oracle-lambda follow-up | Paper scale: gradient-boosted trees for phat_k/mu_k/sigma_k, full paper's covariate-dependent spline lambda with minibatch Adam + early stopping (Appendix C.2), nonlinear DGP variants |
| Hardware | CPU (numpy + torch, no GPU) | CPU/GPU |
| Wall time | Minutes (clf + reg + oracle runs) | Unknown (not attempted) |
| Feasibility | Full for Claim 1; partial for Claim 2 (deviation: parametric fits instead of GBTs, since setting is Linear; fitted lambda pipeline simpler than paper's) |
| Cost | ~$0 | ~$0 |
| Outcome | Claim 1 confirmed; Claim 2 inconclusive (not refuted, not confirmed); Eq. 7 typo side finding |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_e3e25e37db58", "created_at": "2026-07-20T08:32:00+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-20T08:32:00+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #2c3e50;padding-bottom:8px">Multi-Distribution Robust Conformal Prediction: 1 confirmed, 1 inconclusive, 1 side finding</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview Vv4XRZDMM0 &middot; arXiv 2601.02998 &middot; reproduced by kondratevakate &middot; CPU only</div>
  <p style="font-size:13px"><b>Claim 1 (Theorem 1, worst-case validity)</b> &mdash; CONFIRMED, 10 seeds:</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%;margin-bottom:10px">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">method</th><th style="padding:4px">clf worst-cov</th><th style="padding:4px">reg worst-cov</th></tr>
    <tr><td style="padding:4px">MDCP</td><td style="padding:4px;text-align:center;color:#27ae60"><b>0.905</b></td><td style="padding:4px;text-align:center;color:#27ae60"><b>0.909</b></td></tr>
    <tr><td style="padding:4px">max-p baseline</td><td style="padding:4px;text-align:center">0.944 (over-covers)</td><td style="padding:4px;text-align:center">0.939 (over-covers)</td></tr>
    <tr><td style="padding:4px">single-source</td><td style="padding:4px;text-align:center">0.894</td><td style="padding:4px;text-align:center;color:#c0392b">0.327 (collapses)</td></tr>
  </table>
  <p style="font-size:13px"><b>Claim 2 (size reduction vs. max-p, Fig. 2 / Fig. 5)</b> &mdash; INCONCLUSIVE:</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">setting</th><th style="padding:4px">paper claim</th><th style="padding:4px">fitted lambda</th><th style="padding:4px">oracle (constant-lambda)</th></tr>
    <tr><td style="padding:4px">classification</td><td style="padding:4px;text-align:center">34.39%</td><td style="padding:4px;text-align:center;color:#e67e22">4.73% (SE 1.68)</td><td style="padding:4px;text-align:center;color:#e67e22">13.49%</td></tr>
    <tr><td style="padding:4px">regression</td><td style="padding:4px;text-align:center">22.44%</td><td style="padding:4px;text-align:center;color:#e67e22">8.71% (SE 1.85)</td><td style="padding:4px;text-align:center;color:#e67e22">11.91%</td></tr>
  </table>
  <p style="font-size:12px;margin-top:8px">Oracle nearly triples the fitted-lambda classification result &rarr; proves the fitted-lambda optimizer is a real bottleneck within the constant-lambda family; the paper's covariate-dependent family is strictly larger and unbounded by this oracle, so the ceiling is unmeasured. Not refuted, not confirmed.</p>
  <p style="font-size:12px;margin-top:6px"><b>Side finding:</b> Eq. 7's printed indicator orientation gives the complement of the intended prediction set (~90% confidence this is a typesetting error, not a substantive defect) &mdash; standard orientation used for all results above.</p>
</div>
````
