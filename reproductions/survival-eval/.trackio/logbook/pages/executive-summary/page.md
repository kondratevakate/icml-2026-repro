# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ce9e2ae52722", "created_at": "2026-07-19T08:34:37+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-19T08:34:37+00:00"}
-->
Independent from-the-paper reproduction in a fully controlled synthetic version of the paper's ST-vs-OR design, **numpy only, CPU, no GPU, no external data**. **Both claims reproduce.** Claim 1: censoring bias is systematic and **mechanism-dependent, and the dependence lives in the IBS family** — administrative censoring biases the Integrated Brier Score by up to −0.080 while covariate-dependent censoring leaves it essentially untouched; Harrell's C-index instead drifts with the censoring *rate* (+0.027 at 80%) almost identically across mechanisms. Claim 2: with **near-tied models** the censored metric reproduces the oracle ranking in only **29–42% of runs at 80% censoring** (down from 100% at 0%), so the declared 'best model' is usually not the true best. Documented boundary: with widely separated models the ranking survives at 100% everywhere, i.e. the instability is specific to the near-tied regime — consistent with the authors' own rebuttal. Wall time ~15 min; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Both claims, controlled synthetic DGP with exact oracle event times; C-index and IBS families; 3 censoring mechanisms x 5 rates | Semi-synthetic on real clinical datasets (PBC, FLCHAIN, METABRIC), full model panel (DeepHit, NMTLR, parametric, non-parametric), all metric families incl. Uno's C and calibration, CV-corrected t-tests over 225 configurations |
| Hardware | CPU (numpy) | CPU (the paper's setting is also light) |
| Compute time | ~15 min | hours (nested CV over datasets x models x 225 configs) |
| Cost | ~$0 | ~$0 |
| Outcome | Claims 1-2 reproduced; mechanism-dependence localised to IBS; ranking instability confirmed in the near-tied regime | — |

Authors' code: https://github.com/Ghanem01/When-Can-We-Trust-Survival-Model-Evaluation


---
<!-- trackio-cell
{"type": "figure", "id": "cell_794e8118ceec", "created_at": "2026-07-19T08:34:37+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-19T08:34:37+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #7d3c98;padding-bottom:8px">When can we trust survival model evaluation? Censoring bends the metrics</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview Y9gsOEdaNE &middot; reproduced by kondratevakate &middot; CPU, synthetic, exact oracle</div>
  <p style="font-size:13px"><b>Claim 1</b> — metric bias (censored minus oracle) at 80% censoring:</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">mechanism</th><th style="padding:4px">C-index bias</th><th style="padding:4px">IBS bias</th></tr>
    <tr><td style="padding:4px">administrative</td><td style="padding:4px;text-align:center">+0.027</td><td style="padding:4px;text-align:center;color:#c0392b"><b>-0.080</b></td></tr>
    <tr><td style="padding:4px">independent</td><td style="padding:4px;text-align:center">+0.030</td><td style="padding:4px;text-align:center">-0.032</td></tr>
    <tr><td style="padding:4px">covariate-dependent</td><td style="padding:4px;text-align:center">+0.030</td><td style="padding:4px;text-align:center;color:#27ae60">0.000</td></tr>
  </table>
  <p style="font-size:13px;margin-top:10px"><b>Claim 2</b> — how often the censored metric reproduces the oracle ranking of <i>near-tied</i> models:</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">censoring</th><th style="padding:4px">0%</th><th style="padding:4px">20%</th><th style="padding:4px">40%</th><th style="padding:4px">60%</th><th style="padding:4px">80%</th></tr>
    <tr><td style="padding:4px">administrative</td><td style="padding:4px;text-align:center">1.00</td><td style="padding:4px;text-align:center">0.88</td><td style="padding:4px;text-align:center">0.79</td><td style="padding:4px;text-align:center">0.71</td><td style="padding:4px;text-align:center;color:#c0392b"><b>0.33</b></td></tr>
    <tr><td style="padding:4px">independent</td><td style="padding:4px;text-align:center">1.00</td><td style="padding:4px;text-align:center">0.88</td><td style="padding:4px;text-align:center">0.79</td><td style="padding:4px;text-align:center">0.62</td><td style="padding:4px;text-align:center;color:#c0392b"><b>0.29</b></td></tr>
    <tr><td style="padding:4px">covariate-dependent</td><td style="padding:4px;text-align:center">1.00</td><td style="padding:4px;text-align:center">0.88</td><td style="padding:4px;text-align:center">0.88</td><td style="padding:4px;text-align:center">0.62</td><td style="padding:4px;text-align:center;color:#c0392b"><b>0.42</b></td></tr>
  </table>
  <p style="font-size:12px;color:#555;margin-top:8px">Clinical reading: under heavy censoring the "best" survival model chosen by a standard benchmark is the true best only about a third of the time.</p>
</div>
````
