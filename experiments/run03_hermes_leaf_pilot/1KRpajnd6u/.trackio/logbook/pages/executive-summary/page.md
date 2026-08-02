# Executive summary

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d046cbbf5500", "created_at": "2026-08-02T04:00:00+00:00", "title": "Executive summary"}
-->
Independent from-the-paper numerical audit of FluxNet's update rule and head parameterizations, **numpy + torch, CPU, no GPU**. **Propositions 1-3 verified** (conservation, L/U-head bounds, with mutation tests). Claim 6 verified (DCL enforces dual bound, not the architecture). Claims 3, 4, 5 (trained-model accuracy, GPU speedup) left **inconclusive** with documented reasons — no toy substitute produced. Wall time ~26 min; cost ~$0.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Props 1-3 + D-head (update rule & heads, random head outputs) | Also trained FluxNet accuracy on SWE/traffic/spinodal data |
| Hardware | CPU (numpy + torch MLP) | GPU for trained models |
| Compute time | ~26 min | hours across datasets |
| Cost | ~$0 | ~$0 |
| Outcome | Claims 1,2,6 reproduced; 3,4,5 inconclusive (data/training/GPU blockers) | — |

---
<!-- trackio-cell
{"type": "figure", "id": "cell_79014160b563", "created_at": "2026-08-02T04:00:00+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-08-02T04:00:00+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #1f6f8b;padding-bottom:8px">FluxNet: capacity-constrained local transport operators</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview 1KRpajnd6u &middot; arXiv 2602.01941 &middot; reproduced by kondratevate &middot; CPU</div>
  <p style="font-size:13px">Independent numerical audit of the flux-form update and head parameterizations (Propositions 1-3 verified; D-head dual bound attributed to DCL, not architecture). Claims 3-5 (trained-model accuracy / GPU timing) left inconclusive with reasons.</p>
  <p style="font-size:12px;color:#555">Why it matters: flux-form surrogates promise conservative, bounded transport on CPU — relevant to surrogate modelling where mass/charge conservation is non-negotiable.</p>
</div>
````
