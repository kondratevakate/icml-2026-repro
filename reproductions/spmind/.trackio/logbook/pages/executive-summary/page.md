# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0dc76011e350", "created_at": "2026-07-29T17:06:36+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T17:06:36+00:00"}
-->
**Outcome.** Five SP-Mind claims were audited without
leaderboard or peer-reproduction evidence. The released agent wiring is
**PARTIALLY VERIFIED**, and the complete SP-Bench manifest is **VERIFIED**.
Three numerical performance claims remain **INCONCLUSIVE** because the
original model, predictions, traces, or evaluation inputs are unavailable.

## Scope & cost

| Item | Value |
| --- | --- |
| Prepared score | 3/10 |
| GPU | None |
| LLM/API calls | None |
| Full dataset download | Not required |
| Unit tests | 8 passing |
| Author commit | `d5b889c3649fbdfd1489e5b2f60fc52a0d3ddbc6` |

Provenance: [arXiv](https://arxiv.org/abs/2606.24235),
[author repository](https://github.com/tomtommyyuan/spmind), and the official
[dataset](https://huggingface.co/datasets/tomyuanyucheng/spmind).


---
<!-- trackio-cell
{"type": "figure", "id": "cell_6a31f59a9daa", "created_at": "2026-07-29T17:06:36+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T17:06:36+00:00"}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">SP-Mind: independent artifact audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | UJcB3XrffF | arXiv 2606.24235 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">Agent architecture</td><td>Wiring verified; end-to-end execution not run</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">SP-Bench accuracy</td><td>Retired model and original traces unavailable</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">Benchmark construction</td><td>102 unique tasks, 18 categories, 8 stages</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">CRC-CODEX</td><td>Required input/output bundle absent</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">Cell annotation</td><td>Ground truth present; predictions absent</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 3/10. No peer reproductions used.</p></div>
````
