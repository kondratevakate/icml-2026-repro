# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_698fb1807e7a", "created_at": "2026-07-23T09:12:58+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-23T09:12:58+00:00"}
-->
**FunCQNet** (ICML 2026 poster `wNL0qGb2MN`) proposes a functional censored quantile neural network for post-transplant kidney survival, evaluated on simulations and on the **UNOS/OPTN** transplant registry. This reproduction targets the paper's **structural core claim** from the simulation study (Appendix D.1) on CPU, because (a) no code was released and (b) the real-data results depend on the access-controlled UNOS/OPTN STAR files (DUA-gated), which are not publicly obtainable. Using a transparent linear-in-Z B-spline quantile estimator with IPCW censoring correction — not the neural network — we **reproduce the core claim**: when the true effect is Z-modulated (Scenarios S1/S2), an interaction-aware model cuts coefficient-function RMSE by **58–85%** versus a Z-agnostic reduced model, and the α₂ reduction (~75%) matches the paper's reported ~75% (Table 3, τ=0.5), holding across 0–50% censoring. The paper's **S3 "comparable when Z is absent" claim does not hold for the unregularized estimator** (interaction ~2× worse) but is **recovered by adding an L1 penalty**, showing S3 comparability depends on FunCQNet's built-in regularization. The **real-data claims are not reproducible** (data private).

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Simulation core claim S1/S2/S3, τ=0.5, CPU estimator | FunCQNet NN + conformal inference + UNOS/OPTN real data |
| Hardware | CPU (laptop) | GPU (NN over n=50000 × B=100 × 3 scen × 4 cens × multi-τ) |
| Compute time | ~1 min (180 fits) | days |
| Cost | $0 | GPU-hours + UNOS/OPTN DUA (weeks–months, fee) |
| Outcome | Core claim REPRODUCED; S3 needs regularization; real-data NOT REPRODUCIBLE | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_15a5cb1b91c7", "created_at": "2026-07-23T09:12:58+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-23T09:12:58+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,sans-serif;max-width:900px;margin:auto;border:1px solid #ddd;border-radius:12px;padding:24px;background:#fafafa;color:#1a1a1a">
  <h2 style="margin:0 0 4px">Reproduction: FunCQNet — Functional Censored Quantile Neural Network</h2>
  <p style="margin:0 0 16px;color:#555">ICML 2026 poster wNL0qGb2MN · kidney transplant survival · CPU core-claim reproduction</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div style="background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:14px">
      <h3 style="margin:0 0 8px">Core claim (S1/S2) — REPRODUCED</h3>
      <p style="margin:0">Interaction model cuts coefficient-function RMSE by <b>58–85%</b> vs a Z-agnostic model when the true effect is Z-modulated. α₂ reduction ~75% matches the paper's ~75% (Table 3, τ=0.5). Holds across 0–50% censoring.</p>
    </div>
    <div style="background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:14px">
      <h3 style="margin:0 0 8px">S3 "comparable" — needs regularization</h3>
      <p style="margin:0">Unregularized interaction model is ~2× worse when Z is irrelevant; an L1 penalty on Z-interaction terms restores comparability (0.076→0.056) while keeping S1 gains — the paper's S3 result hinges on FunCQNet's dropout.</p>
    </div>
    <div style="background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:14px">
      <h3 style="margin:0 0 8px">Real data (UNOS/OPTN) — NOT REPRODUCIBLE</h3>
      <p style="margin:0">Access-controlled STAR files, DUA-gated, no public version. No reference code released.</p>
    </div>
    <div style="background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:14px">
      <h3 style="margin:0 0 8px">Method</h3>
      <p style="margin:0">Linear-in-Z B-spline quantile regression + IPCW (Bang–Tsiatis) median regression via LP. Faithful DGP; true α is linear in Z so the estimator can represent it exactly — a fair, not strawman, comparison.</p>
    </div>
  </div>
</div>
````
