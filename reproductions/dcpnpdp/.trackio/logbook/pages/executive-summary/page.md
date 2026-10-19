# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5976d7201f39", "created_at": "2026-07-20T08:31:34+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-20T08:31:34+00:00"}
-->
Independent from-the-paper reproduction of DC-PnPDP's core theory claims, on a known synthetic subspace manifold with an exact proximal denoiser (**numpy, CPU only, no GPU, no medical data**). **Claim 1 (dual-variable coupling removes steady-state bias) reproduces cleanly across 12 seeds with zero overlap between the dual-coupled and loose-coupled (u=0) schemes at every softness level s.** **Claim 2's underlying fixed-point theorem is correct, but its "formalizes the asymptotic convergence guarantee" framing overstates what Theorem C.1 delivers** — a precise, sourced correction (traceable to the paper's own text), not a vague hedge. Although DC-PnPDP targets medical image reconstruction (CT/MRI), the claim under test here is a general PnP/diffusion-prior theory point, fully reproducible with zero real medical data (synthetic manifold + exact denoiser, controlled and analytic). Total cost ~$0, wall time under a minute.

## Scope & cost

| Item | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Both claims on a known r=8-dim linear subspace of R^60, exact proximal shrink denoiser (softness s in {0.5, 0.1, 0.01, 0}), dual-coupled vs. loose-coupled (u≡0) ADMM backbone, 12 seeds, 4000 iterations | Paper-scale CT/MRI reconstruction with a real stochastic diffusion prior, real measurement operators, non-convex analysis |
| Hardware | CPU (numpy only) | GPU (diffusion sampling) |
| Wall time | < 1 minute | hours |
| Cost | ~$0 | GPU-hours |
| Outcome | Claim 1 cleanly confirmed (12/12 seeds, no overlap); Claim 2 fixed-point characterization confirmed, "asymptotic convergence" framing not delivered by the theorem as stated | — |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_22c175a986cf", "created_at": "2026-07-20T08:31:34+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-20T08:31:34+00:00"}
-->
````html
<!-- poster_embed.html -->
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ddd;border-radius:10px;padding:20px;background:#fafafa;color:#111">
  <h2 style="margin:0 0 4px;font-size:19px;border-bottom:2px solid #2c3e50;padding-bottom:8px">DC-PnPDP: dual-variable coupling removes PnP steady-state bias; convergence framing overstated</h2>
  <div style="font-size:12px;color:#555;margin-bottom:10px">ICML 2026 &middot; OpenReview jEBkuuETjr &middot; arXiv 2602.23214 &middot; reproduced by kondratevakate &middot; CPU only, no medical data</div>
  <p style="font-size:13px"><b>Claim 1</b> (Sec. 3) &mdash; median distance-to-manifold at convergence, dual-coupled vs. loose (u&equiv;0), 12 seeds, no overlap at any s:</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eee"><th style="padding:4px;text-align:left">softness s</th><th style="padding:4px">dual-coupled</th><th style="padding:4px">loose (u=0)</th></tr>
    <tr><td style="padding:4px">0.5</td><td style="padding:4px;text-align:center;color:#27ae60">9.6e-03</td><td style="padding:4px;text-align:center;color:#c0392b">1.3e-02</td></tr>
    <tr><td style="padding:4px">0.1</td><td style="padding:4px;text-align:center;color:#27ae60">2.3e-03</td><td style="padding:4px;text-align:center;color:#c0392b">1.0e-02</td></tr>
    <tr><td style="padding:4px">0.01</td><td style="padding:4px;text-align:center;color:#27ae60">2.5e-04</td><td style="padding:4px;text-align:center;color:#c0392b">9.7e-03</td></tr>
    <tr><td style="padding:4px">0 (idealized)</td><td style="padding:4px;text-align:center;color:#27ae60"><b>3.9e-16</b></td><td style="padding:4px;text-align:center;color:#c0392b">9.6e-03</td></tr>
  </table>
  <p style="font-size:13px;margin-top:8px"><b>Claim 2</b> (Thm. C.1) &mdash; fixed-point optimality characterization is real and reproducible, but "formalizes the asymptotic convergence guarantee" overstates it: no convergence dynamics are proven (paper concedes global convergence for non-convex PnP is "an open theoretical challenge"), and fixed points land on the data manifold only in the degenerate limit &gamma;&rarr;&infin;, outside Assumption D.1's finite-&gamma; regime &mdash; visible above as nonzero distance for every non-degenerate s&gt;0.</p>
</div>
````
