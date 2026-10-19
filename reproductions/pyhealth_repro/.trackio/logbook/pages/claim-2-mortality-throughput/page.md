# Claim 2: mortality throughput

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pyhealth_c2_001", "created_at": "2026-07-29T14:30:00+00:00", "title": "Claim 2: mortality throughput"}
-->
**Anchored claim (verbatim).** "On MIMIC-IV in-hospital mortality prediction,
PyHealth 2.0 with 16 workers processes data in 2,385 seconds versus 93,708
seconds for a naive pandas implementation, a roughly 39x speedup, while
reducing peak memory from 146.23-457.42 GB (PyHealth 1.16) to 17.48-28.70 GB,
about a 20x reduction (Table 3)."

**Verdict -- NOT ATTEMPTED.**

The paper benchmarks MIMIC-IV v2.2 on an AMD EPYC 7513 workstation with 1 TB
RAM. A full reproduction requires the complete dataset, the mortality task,
worker counts 1/4/8/12/16, peak-memory instrumentation, and the Pandas,
PyHealth 1.16, MEDS, and PyHealth 2.0 baselines.

The public MIMIC demo and the local one-worker smoke path cannot test the
claimed full-data runtime, scaling, or memory ratios. No toy result is
substituted for this claim.
