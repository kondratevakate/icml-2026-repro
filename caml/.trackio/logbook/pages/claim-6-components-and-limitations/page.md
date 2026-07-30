# Claim 6: components and limitations


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_42cac13fbf1b", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 6: components and limitations"}
-->
**PARTIAL - 1/2.** Offset and gate component mechanics are
independently checked. No ablation, schedule-sensitivity, optimizer, benign
failure-case, or overhead runner is released.


---
<!-- trackio-cell
{"type": "code", "id": "cell_3069eb2396bc", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 6: components and limitations evidence", "language": "python"}
-->
````output
{
  "aligned_constraint": {
    "seed": 260525001,
    "expected_c": -0.33486423514651853,
    "official_c": -0.33486423514651853,
    "absolute_difference": 0.0,
    "offset_detached": true,
    "schedule": {
      "0": 0.0,
      "1": 0.0,
      "2": 0.0,
      "3": 0.3333333333333333,
      "4": 0.6666666666666666,
      "5": 1.0,
      "6": 1.0
    },
    "expected_schedule": {
      "0": 0.0,
      "1": 0.0,
      "2": 0.0,
      "3": 0.3333333333333333,
      "4": 0.6666666666666666,
      "5": 1.0,
      "6": 1.0
    },
    "total_is_finite": true
  },
  "limits": {
    "canonical_gpu": "NVIDIA RTX 5090",
    "canonical_five_seed_runs_completed": false,
    "full_table_regenerated": false
  }
}
````
