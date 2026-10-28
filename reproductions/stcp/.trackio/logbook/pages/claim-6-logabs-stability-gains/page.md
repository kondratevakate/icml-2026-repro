# Claim 6: LogAbs stability gains


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5538bc13c9a", "created_at": "2026-07-29T16:17:42+00:00", "title": "Claim 6: LogAbs stability gains"}
-->
**Verdict - PENDING FRESH OFFICIAL SYNTHETIC RUN.**

The pinned author LogAbs pipeline completed a local end-to-end smoke run, but
one repeat and reduced training are insufficient to estimate stability.
The publishable runner labels such output `NO CLAIM VERDICT`; only a fresh
canonical 50-repeat run is eligible for the final claim verdict.

The paper defines Std with denominator `R-1`, whereas the released synthetic
aggregator uses NumPy's population denominator `R`. The runner reports both
the released and sample-corrected values.


---
<!-- trackio-cell
{"type": "code", "id": "cell_bbceb399c023", "created_at": "2026-07-29T16:17:42+00:00", "title": "C6 machine-readable evidence", "language": "python"}
-->
````output
{
  "metric_implementation_note": "The paper defines Std with denominator R-1, while the released synthetic aggregator uses numpy.std with its default denominator R.",
  "metric_mismatch_detected": true,
  "verdict": "PENDING FRESH OFFICIAL SYNTHETIC RUN"
}
````
