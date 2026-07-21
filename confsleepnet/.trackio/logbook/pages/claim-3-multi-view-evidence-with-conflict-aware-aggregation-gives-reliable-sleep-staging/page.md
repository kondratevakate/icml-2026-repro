# Claim 3: Multi-view evidence with conflict-aware aggregation gives reliable sleep staging


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_77ae11621d6a", "created_at": "2026-07-18T17:26:59+00:00", "title": "Claim 3: Multi-view evidence with conflict-aware aggregation gives reliable sleep staging"}
-->
**Status: BLOCKED (not attempted as a full run).**

This is the headline empirical claim (Acc / MF1 tables on SleepEDF-20/78, MASS-SS3, SHHS). Reproducing it needs the full ConfSleepNet pipeline (no upstream code) and the datasets (SleepEDF download pending). The *aggregation mechanism* it relies on is the one verified analytically in Claim 1 (with the Eq. 8 caveat), so the theoretical ingredient is reproduced even though the end-to-end staging numbers are not. Flagged as a blocker; a GPU training run on SleepEDF is the natural next step once the data lands.
