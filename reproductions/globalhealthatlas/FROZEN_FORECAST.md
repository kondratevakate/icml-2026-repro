# Frozen forecast

Frozen at `2026-07-30T00:12:15+04:00`, after the paper/source/repository/Hub
inventory and before any independent aggregation, execution, or numerical
comparison.

## Forecast

**6/12 points**, plausible range **5–8/12**.

Expected claim-level support:

| Claim | Forecast | Basis available before checks |
|---|---:|---|
| C1 corpus scale and coverage | 0/2 | The corpus itself is not among the linked public artifacts. |
| C2 corpus composition and QC | 1/2 | Aggregate counts may be recoverable, but raw corpus and expert audit rows are absent. |
| C3 evaluator validity/stability | 0/2 | Adapter exists, but expert labels and repeated-run prediction matrices are not evident. |
| C4 multilingual benchmark | 2/2 | Detailed per-model cached CSVs and three aggregate tables are released. |
| C5 SFT and transfer | 2/2 | Dedicated ablation, incremental, and transfer CSVs are released. |
| C6 robustness and leakage | 1/2 | Dedicated cached CSVs exist; end-to-end regeneration is likely GPU/API bound. |

Main upside: the released detailed benchmark tables may independently
reconstruct more paper values than their filenames imply.

Main downside: the repository may contain display exports without raw
predictions, labels, evaluator gold annotations, perturbation inputs, or
generation seeds.

