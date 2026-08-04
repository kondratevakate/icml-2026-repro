# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6a35bb2a5b12", "created_at": "2026-07-20T08:32:00+00:00", "title": "Reproduction bundle"}
-->
**Summary.** Theorem 1 (finite-sample worst-case validity) reproduces cleanly and matches the paper's own reported number (90.25%). The prediction-set size-reduction claim (Fig. 2 / Fig. 5) is inconclusive: measured reductions (4.73% classification, 8.71% regression) are far below the paper's claimed 34.39% / 22.44%, but an oracle-lambda follow-up restricted to a strictly smaller search space (constant-lambda vs. the paper's covariate-dependent spline family) already reaches 13.49% / 11.91% — nearly tripling the fitted classification result — which demonstrates the fitted-lambda optimizer used here under-optimizes and leaves the true ceiling of a faithful reproduction unmeasured. This is reported as not refuted / not confirmed, not spun either direction.

**Independently useful side finding.** Eq. 7's printed p-value indicator orientation, taken literally, yields the complement of the intended prediction set. This looks like a one-character typesetting error (confidence ~90%; see Claim 2 for the reasoning) worth flagging to the authors — not a defect in the method itself. All results in this logbook use the standard orientation.

**Reproduction bundle** (all scripts, results, and logs from `C:\Projects\02_academia\icml-repro\mdrcp\`) is attached as the artifact cell below: `spec.md` (transcribed spec), `verify_mdrcp.py` / `run_one.py` / `run_all.py` (main pipeline), `oracle.py` / `oracle_reg.py` (oracle-lambda follow-up), `res_clf.json` / `res_reg.json` / `res_oracle_clf.json` (results), `clf.log` / `reg.log` / `oracle_clf.log` / `oracle_reg.log` (run logs).

**Rerun:**

    python run_one.py          # fitted-lambda classification + regression, 10 seeds
    python oracle.py           # oracle constant-lambda, classification
    python oracle_reg.py       # oracle constant-lambda, regression

Requires numpy + torch (CPU only, no scipy/sklearn). No GPU, no external data (synthetic DGP per `spec.md`).


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_1a74587d33f4", "created_at": "2026-07-20T14:07:35+00:00", "title": "Reproduction bundle", "artifact": "repro-multi-distribution-robust-conformal-prediction/repro-bundle:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-multi-distribution-robust-conformal-prediction/repro-bundle:v0` · dataset

https://huggingface.co/buckets/kondratevakate/repro-multi-distribution-robust-conformal-prediction-artifacts#repro-multi-distribution-robust-conformal-prediction/repro-bundle:v0
