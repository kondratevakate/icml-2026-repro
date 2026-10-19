# Claim 3: main verification performance


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ec0dbc84fd34", "created_at": "2026-07-30T07:15:07+00:00", "title": "Claim 3: main verification performance"}
-->
**UNSUPPORTED - 0/2.** The approximate headline is recoverable
only as table arithmetic: relative to Self-Consistency averaged across six
cells, AUROC rises by
12.25% and
Brier falls by
52.60%.
No row-level predictions are released. Moreover, all five Qwen3
diverticulitis values quoted in the Main Results prose conflict with Table 1.


---
<!-- trackio-cell
{"type": "code", "id": "cell_63a59e3a72c0", "created_at": "2026-07-30T07:15:07+00:00", "title": "Claim 3: main verification performance evidence", "language": "python"}
-->
````output
{
  "headline": {
    "versus_self_consistency": {
      "auroc_relative_gain": 0.12251813101890377,
      "brier_relative_reduction": 0.5259566620562471
    },
    "versus_best_baseline_per_cell": {
      "mean_relative_auroc_gain": 0.09241618580082389,
      "mean_relative_brier_reduction": 0.5038920953193996
    }
  },
  "prose_table_consistency": {
    "context": "Qwen3-30B diverticulitis example",
    "all_values_conflict": true,
    "comparisons": [
      {
        "metric": "K3 AUROC",
        "main_results_prose": 0.9789,
        "table_1": 0.9794,
        "absolute_difference": 0.000500000000000056
      },
      {
        "metric": "K3 Risk@0.5",
        "main_results_prose": 0.0802,
        "table_1": 0.0741,
        "absolute_difference": 0.006099999999999994
      },
      {
        "metric": "K3 Brier",
        "main_results_prose": 0.0647,
        "table_1": 0.0632,
        "absolute_difference": 0.0014999999999999875
      },
      {
        "metric": "Active AUROC",
        "main_results_prose": 0.9856,
        "table_1": 0.9862,
        "absolute_difference": 0.0005999999999999339
      },
      {
        "metric": "Active Risk@0.5",
        "main_results_prose": 0.0494,
        "table_1": 0.037,
        "absolute_difference": 0.012400000000000001
      }
    ]
  }
}
````
