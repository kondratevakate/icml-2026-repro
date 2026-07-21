# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_29fbac77d2ea", "created_at": "2026-07-18T16:28:12+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-18T16:28:12+00:00"}
-->
Independent from-the-paper reproduction of CalPro's uncertainty-quantification claims. **Headline result: anchored claim A2 (Theorem 4.2's coverage guarantee) is REFUTED** by a text/theoretical audit of the paper's own Section 4, cross-checked against the extracted paper text — the bound is not "structure-aware" by the paper's own admission (Sec. 4.3), its shift term contains an unestimable quantity, and a numeric check (`audit_thm42.py`) shows it goes deeply vacuous under realistic shift magnitudes (L_s * epsilon = 39.47, well past 1 - alpha at every tested tau). Full detail on the claim-1-2 page.

**Correction:** an earlier version of this logbook reported a "Claim 1 reproduced" verdict against anchored claim **A4** (X-ray -> cryo-EM modality-shift degradation, "only ~3.8pp vs 14-18pp for baselines"). That verdict is withdrawn: the scripts it was based on tested a self-constructed synthetic covariate-shift regression, not proteins, not pLDDT, and not CalPro's actual method (graph evidential head + differentiable conformal layer + domain priors). **A4's honest status is NOT TESTED**, not confirmed and not refuted. A from-scratch CPU check of A4 on open PDB/AlphaFold data was assessed as technically feasible but not attempted — the RMSD/structural-alignment pipeline has high silent-error risk against an unverifiable target, and any result could not honestly be labeled "CalPro" since the real graph architecture and train split are not released; flagged as future work.

Claim 2 (calibration error, anchored A6) and Claim 3 (docking, anchored A5) are unaffected by this correction. **Claim 2 (calibration error) reproduces** on the paper's own non-biological benchmark (heteroscedastic regression under covariate shift, p.13; numpy + torch, CPU only): ECE cut by **81% in-distribution / 26% under shift**. **Claim 3 (protein docking) is out of scope** — it needs the AlphaFold/docking pipeline. Total wall time ≈ 2 minutes on CPU; cost ≈ $0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | A2 (Theorem 4.2): text/theoretical audit + numeric vacuity check, refuted. A4 (modality-shift degradation): not tested — earlier synthetic-proxy verdict withdrawn. A6 (calibration error): reproduced on the paper's own non-biological benchmark. A5 (docking): skipped | All claims on protein structures (AlphaFold pLDDT, GNN evidential head, ligand docking) across modality / temporal / disorder shifts, including a real numeric check of A4 |
| Hardware | CPU (numpy + torch 2.13.0+cpu) | GPU for the graph evidential head + docking pipeline |
| Compute time | ≈ 2 min | hours–days (protein data + training + docking) |
| Cost | ≈ $0 | GPU + docking compute |
| Outcome | A2 refuted (theorem vacuity); A4 not tested (correction from earlier invalid verdict); A6 (calibration error) reproduced (ECE −81% / −26%); A5 (docking) out of scope | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_d68f25aad929", "created_at": "2026-07-18T16:28:12+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-18T16:28:12+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <div style="border-bottom:2px solid #c0392b;padding-bottom:8px;margin-bottom:12px">
    <h2 style="margin:0;font-size:20px">Reproducing CalPro: Theorem 4.2's coverage guarantee is refuted</h2>
    <div style="font-size:12px;color:#555">ICML 2026 · OpenReview 3LRWjJTp0Y · arXiv 2601.07201 · reproduced by kondratevakate · CPU only</div>
  </div>
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    <div style="flex:1;min-width:240px">
      <h3 style="font-size:14px;color:#c0392b;margin:6px 0">Headline: A2 refuted</h3>
      <p style="font-size:13px;margin:4px 0">Theorem 4.2's "structure-aware" PAC-Bayesian coverage guarantee is contradicted by the paper's own Section 4.3 (bound is <b>prior-agnostic</b>), depends on an unestimable shift radius ε, and goes <b>deeply vacuous</b> at realistic shift magnitudes in a numeric check.</p>
      <h3 style="font-size:14px;color:#c0392b;margin:10px 0 6px">Correction: A4 not tested</h3>
      <p style="font-size:13px;margin:4px 0">An earlier "reproduced" verdict on the X-ray→cryo-EM degradation claim (A4) is withdrawn — it tested a synthetic proxy, not proteins, not pLDDT, not CalPro's real method. Honest status: <b>not tested</b>.</p>
    </div>
    <div style="flex:1;min-width:240px">
      <h3 style="font-size:14px;color:#c0392b;margin:6px 0">Theorem 4.2 vacuity check</h3>
      <table style="font-size:12px;border-collapse:collapse;width:100%">
        <tr style="background:#eee"><th style="padding:4px;text-align:left">quantity</th><th style="padding:4px">value</th></tr>
        <tr><td style="padding:4px">L_s (local Lipschitz, 99th pct)</td><td style="padding:4px;text-align:center">9.82</td></tr>
        <tr><td style="padding:4px">ε (W1, cal → shifted)</td><td style="padding:4px;text-align:center">4.02</td></tr>
        <tr><td style="padding:4px">L_s · ε</td><td style="padding:4px;text-align:center;color:#c0392b"><b>39.47</b></td></tr>
        <tr><td style="padding:4px">bound at τ=0.80/0.90/0.95</td><td style="padding:4px;text-align:center;color:#c0392b"><b>deeply negative (vacuous)</b></td></tr>
      </table>
      <p style="font-size:11px;margin:6px 0;color:#555">Caveat: computed on the repo's synthetic toy setting, not real X-ray/cryo-EM data — shows the bound goes vacuous under moderate shift in general, not its value in the paper's own experiment.</p>
      <p style="font-size:12px;margin:8px 0"><b>A2 ✗ refuted</b> (theorem vacuity) · <b>A4</b> not tested (correction) · <b>A6 ✓</b> calibration error ECE −81% in-dist / −26% shift · <b>A5</b> (docking) out of scope</p>
    </div>
  </div>
</div>
````
