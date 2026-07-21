# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f0fa782b2f13", "created_at": "2026-07-18T17:26:59+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-18T17:26:59+00:00"}
-->
Independent from-the-paper reproduction of ConfSleepNet's **conflict-aware aggregation** theory, **numpy, CPU, seconds** (no GPU). **Claim 1 reproduced analytically**: transcribing the paper's own equations, the aggregation operator's uncertainty behaviour (Propositions 1 and 2) holds when uncertainty is taken consistent with the paper's own normalization constraint (sum_k b + u = 1); and the printed Equation 8 appears to contain a typo — it fails Proposition 2 as the conflict C -> 1 (off by a factor ~2 vs the normalization). **Claims 2 and 3 are empirical and blocked**: the upstream repo is README-only (no code) and the SleepEDF data was still downloading, so a faithful full run is deferred. Wall time ~seconds; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Claim 1: analytic verification of Propositions 1-2 + Eq. 8 check | Claims 2-3: train ConfSleepNet on SleepEDF-20/78, MASS, SHHS and reproduce Acc/MF1 tables |
| Hardware | CPU (numpy, closed-form) | GPU (evidential multi-view DNNs, k-fold CV) |
| Compute time | ~seconds | hours (per dataset, k-fold) |
| Cost | ~$0 | GPU compute |
| Outcome | Claim 1 reproduced (props hold under normalization; Eq. 8 as printed likely a typo); Claims 2-3 blocked (no code + data pending) | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_3ad271e5da26", "created_at": "2026-07-18T17:26:59+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-18T17:26:59+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #6c3483;padding-bottom:8px">ConfSleepNet: reproducing the conflict-aware aggregation theory</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview fve4MEzeSp &middot; arXiv 2605.17021 &middot; reproduced by kondratevakate &middot; CPU, analytic</div>
  <p style="font-size:13px"><b>Claim 1</b> — the CMAM aggregation operator, checked by sweeping conflict C in [0,1] (K=5 sleep stages):</p>
  <ul style="font-size:13px;margin:6px 0">
    <li><b>Proposition 1</b> (C-&gt;0, consistent opinions): combined uncertainty decreases &mdash; <span style="color:#27ae60">holds</span>.</li>
    <li><b>Proposition 2</b> (C-&gt;1, conflicting opinions): combined uncertainty increases &mdash; <span style="color:#27ae60">holds under the paper's own normalization</span> sum_k b + u = 1.</li>
    <li><b>Equation 8 as printed</b>: its C-&gt;1 limit <i>decreases</i> the uncertainty and disagrees with the normalization by ~2x &mdash; <span style="color:#c0392b">likely a typo</span>.</li>
  </ul>
  <p style="font-size:12px">A partial refutation at the level of the printed equation, with the underlying design principle confirmed. Claims 2-3 (empirical, SleepEDF) are blocked: upstream repo is README-only.</p>
</div>
````
