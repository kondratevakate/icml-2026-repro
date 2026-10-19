# Claim 3: drug and LOS throughput

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pyhealth_c3_001", "created_at": "2026-07-29T14:30:00+00:00", "title": "Claim 3: drug and LOS throughput"}
-->
**Anchored claim (verbatim).** "On drug recommendation and length-of-stay
prediction tasks, PyHealth 2.0 still yields substantial speedups over pandas
(4,725s to 569s, and 10,269s to 1,096s respectively), though smaller than the
39x mortality-prediction speedup (Tables 4 and 5)."

**Verdict -- NOT ATTEMPTED.**

This claim requires two complete MIMIC-IV v2.2 pipelines, their exact feature
tables, and the same multi-baseline worker sweep as Claim 2. Neither a reduced
patient cohort nor timing only the final task transform would reproduce the
end-to-end measurements in Tables 4 and 5.

The exact table values were located in the paper source, but reading author
results is not an independent reproduction. Execution is deferred until the
full MIMIC-IV data and persistent GCP machine are available.
