# Executive summary

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pyhealth_exec_001", "created_at": "2026-07-29T14:30:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-29T14:30:00+00:00"}
-->
**Outcome.** We independently audited the public arXiv source and the official
PyHealth `v2.0.1` release. Claim 1 is **VERIFIED**: the paper inventories 22
datasets, 43 tasks, 28 models, and 7 attribution methods, while the pinned
release clears every claimed threshold and exposes all named modality and
method witnesses. Claim 4 is **FALSIFIED as anchored**: its 7/24/51 attribution
to Table 2 conflicts with the paper's actual mortality-task row values
34/27/51. Claims 2 and 3 are not attempted without full MIMIC-IV v2.2; Claim 5
is inconclusive because the historical "400+ members" count has no immutable
public measurement artifact.

## Scope and cost

| Item | Value |
| --- | --- |
| GPU | None |
| Protected clinical data | None |
| Runtime | Under 1 second for audit and validation |
| Official code | `v2.0.1`, commit `ed562121b5bae185b36322c64ce6215c2095dd50` |
| Paper | arXiv `2601.16414v2` |
| Prepared score | 4/10 |

All counts and Table 2 values are machine-parsed rather than manually copied.
The evidence bundle includes four tests, including mutations that remove the
genomics exports and rewrite Table 2 to the challenge values.

---
<!-- trackio-cell
{"type": "figure", "id": "cell_pyhealth_poster_001", "created_at": "2026-07-29T14:30:00+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-07-29T14:30:00+00:00", "poster": true}
-->
````html
<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #d8dde3;padding:20px;background:#fff;color:#17202a">
  <h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #16856b;padding-bottom:8px">PyHealth 2.0: public-source reproducibility audit</h2>
  <p style="font-size:12px;color:#52606d">ICML 2026 | OpenReview gMLVFN9hl8 | arXiv 2601.16414v2 | official v2.0.1</p>
  <table style="font-size:12px;border-collapse:collapse;width:100%">
    <tr style="background:#eef4f2"><th style="padding:6px;text-align:left">Claim</th><th>Independent evidence</th><th>Verdict</th></tr>
    <tr><td style="padding:6px">C1 toolkit coverage</td><td style="text-align:center">22 datasets | 43 tasks | 28 models | 7 methods</td><td style="text-align:center;color:#137a46;font-weight:700">VERIFIED</td></tr>
    <tr><td style="padding:6px">C4 seven-line / Table 2</td><td style="text-align:center">Claim says 7/24/51; Table 2 contains 34/27/51</td><td style="text-align:center;color:#b42318;font-weight:700">FALSIFIED</td></tr>
    <tr><td style="padding:6px">C2-C3 throughput</td><td style="text-align:center">Full MIMIC-IV v2.2 and baseline sweep required</td><td style="text-align:center">NOT ATTEMPTED</td></tr>
    <tr><td style="padding:6px">C5 community</td><td style="text-align:center">184 examples; RHealth public; no historical roster</td><td style="text-align:center">INCONCLUSIVE</td></tr>
  </table>
  <p style="font-size:12px;margin:12px 0 0">Prepared score: <strong>4/10</strong>. Data-free, deterministic, standard-library audit with mutation controls.</p>
</div>
````
