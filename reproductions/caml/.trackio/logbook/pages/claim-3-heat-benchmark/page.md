# Claim 3: Heat benchmark


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_81ad3e5d7397", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 3: Heat benchmark"}
-->
**PARTIAL - 1/2.** A finite 10-epoch Heat smoke run executes after
installing the undocumented `overrides` dependency. The code uses boundary
weight 1 and stop threshold 1e-3, versus 5 and 2e-3 in the paper; therefore it
is not a canonical five-seed reproduction.


---
<!-- trackio-cell
{"type": "code", "id": "cell_a06c9b8bdb08", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 3: Heat benchmark evidence", "language": "python"}
-->
````output
{
  "seed": 999,
  "epochs": 10,
  "l2_at_last_epoch": 0.08415783196687698,
  "reached_target_epoch": -1,
  "positive_cosine_rate": 1.0,
  "wall_seconds": 2.2592973000137135,
  "canonical_table_reproduction": false
}
````
