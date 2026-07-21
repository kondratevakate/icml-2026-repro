# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_df5e8047da54", "created_at": "2026-07-19T07:36:53+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-19T07:36:53+00:00"}
-->
**TOY reproduction on real data.** Trained a small CNN on a 12k-image subset of CIFAR-100 with the **real CIFAR-100N human label noise** (coarse, 20 superclasses, 26.4% mislabeled), CPU, tracking per-sample prediction entropy across 25 epochs. **Claim 1 reproduced in direction:** mislabeled samples keep higher entropy than correctly-labeled ones at every epoch, with the separation peaking mid-training (gap 0.51 at epoch 15) — the memorization-order mechanism (clean learned first, noise memorized later). **Claim 2 (SEI) reproduced as a proxy:** an entropy-trajectory statistic separates mislabeled from clean with AUC ~0.71. **Claim 3 (medical-dataset SOTA) not attempted.** Labeled `toy` — this is a subset + small CNN, not the paper's full medical-dataset scale. Wall time ~7 min of compute; cost ~$0.

## Scope & cost

| Item | This reproduction (toy) | Full replication |
| --- | --- | --- |
| Scope | Claims 1-2 mechanism on CIFAR-100N (12k subset, 20 coarse classes) | Claims 1-3 with the full SEI method + baselines on CIFAR-100N + 3 medical datasets (ISIC, DeepDRiD, ...) |
| Hardware | CPU (small CNN, torch) | GPU (larger backbones, full datasets) |
| Compute time | ~7 min | hours (multiple datasets, baselines) |
| Cost | ~$0 | GPU compute |
| Outcome | Claim 1 direction + Claim 2 SEI-proxy reproduced (toy); Claim 3 not attempted | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_28b2fa5796a2", "created_at": "2026-07-19T07:36:53+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-19T07:36:53+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #b9770e;padding-bottom:8px">Entropy dynamics identify mislabeled images (toy, CIFAR-100N)</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview BUxrIaf7Zc &middot; arXiv 2605.31090 &middot; reproduced by kondratevakate &middot; CPU &middot; <b>toy</b></div>
  <p style="font-size:13px"><b>Claim 1</b> — mean prediction entropy of clean vs mislabeled samples over training (CIFAR-100N, 26.4% noise):</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">epoch</th><th style="padding:4px">1</th><th style="padding:4px">5</th><th style="padding:4px">10</th><th style="padding:4px">15</th><th style="padding:4px">20</th><th style="padding:4px">25</th></tr>
    <tr><td style="padding:4px">clean</td><td style="padding:4px;text-align:center">2.64</td><td style="padding:4px;text-align:center">2.02</td><td style="padding:4px;text-align:center">1.56</td><td style="padding:4px;text-align:center">0.94</td><td style="padding:4px;text-align:center">0.39</td><td style="padding:4px;text-align:center">0.14</td></tr>
    <tr><td style="padding:4px">mislabeled</td><td style="padding:4px;text-align:center">2.73</td><td style="padding:4px;text-align:center">2.27</td><td style="padding:4px;text-align:center">1.96</td><td style="padding:4px;text-align:center">1.45</td><td style="padding:4px;text-align:center">0.70</td><td style="padding:4px;text-align:center">0.26</td></tr>
    <tr style="color:#b9770e"><td style="padding:4px"><b>gap</b></td><td style="padding:4px;text-align:center">0.09</td><td style="padding:4px;text-align:center">0.25</td><td style="padding:4px;text-align:center">0.40</td><td style="padding:4px;text-align:center"><b>0.51</b></td><td style="padding:4px;text-align:center">0.31</td><td style="padding:4px;text-align:center">0.12</td></tr>
  </table>
  <p style="font-size:13px;margin-top:8px">Mislabeled entropy stays higher throughout (separation peaks mid-training, then both collapse as the CNN memorizes noise). <b>Claim 2</b>: entropy-trajectory statistic separates mislabeled from clean, <b>AUC ~0.71</b>.</p>
</div>
````
