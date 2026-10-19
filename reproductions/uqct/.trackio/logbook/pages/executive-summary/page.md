# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a1b2437e3eb7", "created_at": "2026-07-18T17:24:07+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-18T17:24:07+00:00"}
-->
Independent from-the-paper reproduction of the paper's **coverage guarantee** (Sequential Likelihood Mixing, Prop. 3.2 / Thm 3.1) on a synthetic phantom — numpy, **CPU only, no LIDC data, no trained checkpoints** (both are 'available on request', not yet released). The SLM confidence sequence attains its nominal (1-delta) coverage of the true image on the **classical single-point reconstruction path** (empirical coverage 1.000 at delta=0.1 and 0.05, 300 runs each). This reproduces the coverage half of Claim 1 and the classical half of Claim 2. The 'deep reconstructions give substantially tighter regions' half needs their U-Net/diffusion checkpoints or GPU training and is left as a GPU phase. Wall time ~5 min; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Coverage guarantee on synthetic phantom, classical (FBP/MLE-style single-point) path | Coverage + tightness on LIDC-IDRI, all reconstructors (FBP, MLE, U-Net, ensembles, diffusion) |
| Hardware | CPU (numpy: Radon + Beer-Lambert/Poisson + SLM) | GPU for U-Net/diffusion training + inference |
| Compute time | ~5 min | hours-days |
| Cost | ~$0 | GPU compute |
| Outcome | Coverage guarantee reproduced on classical path; deep-tightness comparison out of scope | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_d29af40fe917", "created_at": "2026-07-18T17:24:07+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-18T17:24:07+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #16607a;padding-bottom:8px">Principled Confidence CT: reproducing the coverage guarantee (classical path)</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview a6vCNSBBeq &middot; arXiv 2602.05812 &middot; reproduced by kondratevakate &middot; CPU, synthetic phantom</div>
  <p style="font-size:13px">Beer-Lambert forward model + Poisson noise + parallel-beam Radon; Sequential Likelihood Mixing confidence sequence C_t (Prop. 3.2). Empirical coverage of the true image x* (300 runs each):</p>
  <table style="font-size:13px;border-collapse:collapse;width:60%;margin:6px auto">
    <tr style="background:#eee"><th style="padding:5px">delta</th><th style="padding:5px">nominal 1-delta</th><th style="padding:5px">empirical</th></tr>
    <tr><td style="padding:5px;text-align:center">0.10</td><td style="padding:5px;text-align:center">0.900</td><td style="padding:5px;text-align:center;color:#27ae60"><b>1.000</b></td></tr>
    <tr><td style="padding:5px;text-align:center">0.05</td><td style="padding:5px;text-align:center">0.950</td><td style="padding:5px;text-align:center;color:#27ae60"><b>1.000</b></td></tr>
  </table>
  <p style="font-size:12px;margin-top:8px">Coverage guarantee holds (conservative — classical single-point recon gives loose, valid regions). The paper's point that <b>deep</b> reconstructions make these regions substantially tighter is the remaining GPU half.</p>
</div>
````
