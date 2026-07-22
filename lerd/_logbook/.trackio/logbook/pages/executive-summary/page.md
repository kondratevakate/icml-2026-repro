# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_01ff2f51b9c1", "created_at": "2026-07-22T12:42:40+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-22T12:42:40+00:00"}
-->
Two of LERD's claims reproduce independently on commodity CPU. **Claim 3 (toy):**
the reproduced latent event-rate estimate matches LERD's reported "Median Rate"
within 2-3.4% (7.38 vs 7.53 at [5,10] Hz; 18.20 vs 18.84 at [15,20] Hz), and a
trajectory-only neural-ODE baseline scores boundary IoU ~0 at high CS (~0.99),
reproducing Table 1's "high CS, IoU=0". **Claim 6 (real data):** on all 88 subjects
of Cohort A (OpenNeuro ds004504), the AD spectral-slowing signature the claim rests
on holds - AD central frequency is below HC in 19/19 channels, with theta elevated
and alpha reduced. **Claim 4** (75% accuracy) is compute-bound: the faithful pipeline
runs but needs a GPU (~19 h/CPU for the No-prior variant alone). **Claim 5** is not
evaluable - Cohort B data is not public. LERD's own trained-model magnitudes (IoU
0.202-0.473, dLIF frequencies, 75% accuracy) are out of scope on CPU. Total cost:
about 20 min CPU analysis plus a one-time 19 min dataset download; $0.

## Scope & cost

|  | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Claim 3 latent-rate + baseline IoU=0 (toy); Claim 6 AD-slowing on real EEG (19/19 channels) | Train full LERD (EPDE+MELP+dLIF+ERG), reproduce IoU 0.202-0.473 and 75.03% accuracy on both cohorts |
| Hardware | 8-core CPU (no GPU) | GPU (L4 sufficient) |
| Compute time | ~20 min analysis + ~19 min download | ~20-40 min (No-prior) to ~2-8 h (full model) per cohort |
| Cost | $0 | ~$2-16 (est. GPU-h x L4 rate) |
| Outcome | Claims 3 and 6 reproduced; 4 GPU-bound; 5 data-unavailable | not attempted |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_359b49c48ab5", "created_at": "2026-07-22T12:42:40+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-22T12:42:40+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#fff;color:#111;border:1px solid #ddd;border-radius:10px;line-height:1.45">
  <div style="border-bottom:3px solid #2b6cb0;padding-bottom:10px;margin-bottom:14px">
    <div style="font-size:22px;font-weight:700;color:#1a365d">Reproduction: LERD — Latent Event-Relational Dynamics for Neurodegenerative Classification</div>
    <div style="font-size:13px;color:#555;margin-top:4px">OpenReview B5DAV1EA8Z · arXiv 2602.18195 · independent reproduction · CPU-only, $0</div>
  </div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px">
    <div style="flex:1;min-width:170px;background:#f0fff4;border:1px solid #9ae6b4;border-radius:8px;padding:10px">
      <div style="font-weight:700;color:#22543d">Claim 3 (toy) — REPRODUCED</div>
      <div style="font-size:13px;margin-top:4px">Latent rate recovered within 2-3.4%; baseline neural-ODE IoU ~0 at CS ~0.99.</div>
    </div>
    <div style="flex:1;min-width:170px;background:#f0fff4;border:1px solid #9ae6b4;border-radius:8px;padding:10px">
      <div style="font-weight:700;color:#22543d">Claim 6 (real data) — REPRODUCED</div>
      <div style="font-size:13px;margin-top:4px">AD spectral slowing vs HC in 19/19 channels (all 88 subjects, ds004504).</div>
    </div>
    <div style="flex:1;min-width:170px;background:#fffaf0;border:1px solid #f6ad55;border-radius:8px;padding:10px">
      <div style="font-weight:700;color:#7b341e">Claim 4 — GPU-bound · Claim 5 — data private</div>
      <div style="font-size:13px;margin-top:4px">Pipeline built &amp; smoke-tested; ~19 h/CPU. Cohort B not public.</div>
    </div>
  </div>

  <div style="font-weight:700;color:#1a365d;margin:8px 0 4px">Claim 3: latent event-rate recovery (Table 1 "Median Rate")</div>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr style="background:#ebf4ff"><th style="text-align:left;padding:5px;border:1px solid #cbd5e0">Band (Hz)</th><th style="padding:5px;border:1px solid #cbd5e0">Reproduced median [95% CI]</th><th style="padding:5px;border:1px solid #cbd5e0">Paper LERD</th><th style="padding:5px;border:1px solid #cbd5e0">GT</th><th style="padding:5px;border:1px solid #cbd5e0">Baselines</th></tr>
    <tr><td style="padding:5px;border:1px solid #cbd5e0">[5,10]</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">7.38 [4.50, 12.83]</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">7.53 [4.30, 14.87]</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">7.5</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">1.000 / 0.340</td></tr>
    <tr style="background:#f7fafc"><td style="padding:5px;border:1px solid #cbd5e0">[15,20]</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">18.20 [11.55, 30.72]</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">18.84 [10.47, 35.24]</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">17.5</td><td style="padding:5px;border:1px solid #cbd5e0;text-align:center">far from GT</td></tr>
  </table>

  <div style="font-weight:700;color:#1a365d;margin:12px 0 4px">Claim 6: AD spectral slowing on Cohort A (ds004504, 88 subjects)</div>
  <div style="font-size:13px">AD central frequency (4-12 Hz centroid) &lt; HC in <b>19/19 channels</b> (paper's model-derived: 18/19). Relative theta: AD 0.062 &gt; HC 0.055; relative alpha: AD 0.029 &lt; HC 0.048; FTD intermediate. Raw centroids (~6.4-8.0 Hz) fall in the paper's Figure 2 range (~6.6-7.8 Hz).</div>

  <div style="margin-top:12px;font-size:12px;color:#666;border-top:1px solid #eee;padding-top:8px">
    Scope: LERD's trained-model magnitudes (IoU 0.202-0.473, dLIF frequencies, 75% accuracy) are out of scope on CPU. Data via anonymous-S3 MCP server (mcp_openneuro). numpy / mne / scipy, deterministic. Full poster embed: poster_embed.html.
  </div>
</div>
````
