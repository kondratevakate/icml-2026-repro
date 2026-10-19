# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ccd_v2_conclusion", "created_at": "2026-07-29T09:00:00+00:00", "title": "Conclusion"}
-->
All six judge-anchored claims have direct evidence. C1, C3, C4, and C6 use
independent calculations or experiments. C2 and C5 use exact, hash-pinned
released outputs and reproduce the authors' aggregation without claiming a
fresh model-training run.

The main practical finding remains unchanged: marginal coverage can conceal
large local errors. On the paper's heteroscedastic model, learned ERT recovers
92% of the analytic violation while an unaligned grouping recovers only 5%.

Rerun:

```powershell
cd repro_ccd_v2
python run_all.py
```
