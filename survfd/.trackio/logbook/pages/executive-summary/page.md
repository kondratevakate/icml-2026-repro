# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ef0b305e5495", "created_at": "2026-07-19T09:09:07+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-19T09:09:07+00:00"}
-->
Independent from-the-paper reproduction, **numpy only, CPU, no GPU, no external data**. **Claim 1 reproduced exactly and seed-robustly**: on the paper's own Theorem 3.2 (independent features, G either linear-including-interactions or purely-additive), SurvFD recovered the ground-truth time-dependent/time-independent partition **exactly in 3/3 test scenarios**, holding on **12/12 independent seeds** with a 15-order-of-magnitude margin between time-dependent and time-independent effect spreads. A first version of this script mistakenly redrew Monte-Carlo reference samples independently at each timepoint, which injected noise indistinguishable from time-dependence -- fixed by common random numbers. **Claim 2 reproduced cleanly, correctly stated**: the n-Shapley interaction index is **1.96x larger** (mean 1.9591, sd 0.0032 across 12 seeds) in scenarios with a true interaction than without; the no-interaction baseline is **not** zero but converges to a stable non-zero constant (0.0284) because a log-hazard-additive model is provably non-additive on the survival scale after the `exp(.)` link -- this is the correct null, not Monte-Carlo error, and the original "directional, not exact" framing understated a precise, reproducible result. **Claim 3 (Theorem 3.3, asymmetric propagation) confirmed 10/10 seeds**: on the paper's own worked example (App. A.2), a genuine time-dependent interaction propagates down to every subset containing it (spread 2.818) but never up to a superset outside it (spread 1.28e-15, machine zero) -- a 15-order-of-magnitude asymmetry, with the paper's closed-form solution matched to within MC error. **Claim 4 (local accuracy, Fig. 2's ten simulated scenarios) confirmed at machine precision**: average local-accuracy error sigma-bar is of order 1e-16/1e-17 on the log-hazard and survival scales, far below the paper's own <1e-5/<1e-3 targets, across all ten scenarios and multiple seeds -- honestly scaled down from the paper's dense time grid (paper's <0.015 threshold specifically concerns *predicted* survival from fitted CoxPH/GBSA models, which were **not** attempted here, no sklearn/scikit-survival available). Reproducibility note: the paper's Eq. (10) does not itself specify the lower-order-term construction; the literal reading breaks local accuracy on the six scenarios with real feature interactions, and the correct construction has to be derived from the Möbius transform given in Appendix A.6. Wall time ~15 min; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Claims 1-2: 2-feature synthetic scenarios spanning the paper's taxonomy, exact brute-force decomposition, 12-seed robustness sweep. Claims 3-4: the paper's own Appendix A.2 worked example and Fig. 2's ten simulated scenarios (p=3, n=1000), exact order-2 SurvSHAP-IQ via the corrected Möbius construction | The paper's 10 simulation scenarios plus real survival datasets (CoxPH/GBSA models), local interpretation case studies |
| Hardware | CPU (numpy, Monte-Carlo marginal integration) | CPU (method is not GPU-bound) |
| Compute time | ~15 min across all 4 claims and seed sweeps | tens of minutes across scenarios/datasets |
| Cost | ~$0 | ~$0 |
| Outcome | Claim 1: exact recovery per Theorem 3.2, 12/12 seeds. Claim 2: 1.96x interaction ratio, seed-stable, correct non-zero null identified. Claim 3: Theorem 3.3 asymmetric propagation, 10/10 seeds. Claim 4: local accuracy at machine precision, 10/10 scenarios (ground-truth functions only; fitted-model version not attempted) | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_e65a9b7297e7", "created_at": "2026-07-19T09:09:07+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-19T09:09:07+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:18px;border-bottom:2px solid #117864;padding-bottom:8px">SurvFD / SurvSHAP-IQ: decomposing survival model interactions in time</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview SldP4LGjdz &middot; arXiv 2602.16505 &middot; reproduced by kondratevakate &middot; CPU, synthetic, exact</div>
  <p style="font-size:13px"><b>Claim 1</b> (Theorem 3.2 exact recovery, 12/12 seeds, common-random-number MC):</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">scenario</th><th style="padding:4px">ground truth</th><th style="padding:4px">SurvFD</th><th style="padding:4px">match</th></tr>
    <tr><td style="padding:4px">A additive, TI</td><td style="padding:4px;text-align:center">x1 TI, x2 TI</td><td style="padding:4px;text-align:center">x1 TI, x2 TI</td><td style="padding:4px;text-align:center;color:#27ae60">&#10003;</td></tr>
    <tr><td style="padding:4px">B additive, x2 TD</td><td style="padding:4px;text-align:center">x1 TI, x2 TD</td><td style="padding:4px;text-align:center">x1 TI, x2 TD</td><td style="padding:4px;text-align:center;color:#27ae60">&#10003;</td></tr>
    <tr><td style="padding:4px">C linear + interaction</td><td style="padding:4px;text-align:center">x1x2 TI</td><td style="padding:4px;text-align:center">x1x2 TI</td><td style="padding:4px;text-align:center;color:#27ae60">&#10003;</td></tr>
  </table>
  <p style="font-size:13px;margin-top:8px"><b>Claim 2</b>: n-Shapley interaction index, mean over t=0.5/2/5, 12-seed sweep:</p>
  <p style="font-size:13px">no interaction (A,B): <b>0.0284</b> (correct non-zero null, not MC error) &nbsp;|&nbsp; with interaction (C,D): <b>0.0557</b> &nbsp;=&nbsp; <b>1.96x</b> ratio (sd 0.0032)</p>
  <p style="font-size:13px;margin-top:8px"><b>Claim 3</b> (Theorem 3.3 asymmetric propagation, paper's App. A.2 example, 10/10 seeds):</p>
  <p style="font-size:13px">downward spread (contaminated subset): <b>2.818</b> &nbsp;|&nbsp; upward spread (protected superset): <b>1.28e-15</b> (machine zero)</p>
  <p style="font-size:13px;margin-top:8px"><b>Claim 4</b> (local accuracy, Fig. 2's ten scenarios, paper target &lt;1e-5 log-hazard / &lt;1e-3 survival):</p>
  <p style="font-size:13px">measured sigma-bar: order <b>1e-16 to 1e-17</b> on both scales, all 10 scenarios &mdash; machine precision, no fitted-model version attempted</p>
  <p style="font-size:12px;color:#555">Directly usable: separating time-dependent from time-independent survival-model effects, with a provably one-directional propagation rule, is exactly what a scanner-effect / disease-progression interpretability workflow needs.</p>
</div>
````
