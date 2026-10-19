# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_71b4863ac243", "created_at": "2026-07-29T20:21:46+00:00", "title": "Conclusion"}
-->
The released simulator and the scoped OOD mechanism are
reproducible on CPU. The paper's epsilon-margin theorem is correct under its
explicit reliability assumption. The broad dynamics and shielding tables are
not reproduced, and the released predictive-shield path requires explicit
repairs before an end-to-end rerun.

## Evidence boundary

- No paper table is treated as execution evidence.
- The CPO run is one algorithm, cohort, and training seed.
- The theorem is conditional and does not certify BA-NODE.
- Source divergences limit reproducibility but do not falsify clinical gains.
