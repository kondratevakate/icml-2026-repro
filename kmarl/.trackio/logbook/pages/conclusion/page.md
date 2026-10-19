# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_81c16f47eb93", "created_at": "2026-07-20T08:30:50+00:00", "title": "Reproduction bundle"}
-->
**Summary.** Five of six anchored claims now have direct evidence: Theorems 4.1 and 4.2 pass independent derivation and finite-sample bound audits; Theorems 4.3/4.4 and Eq. 11 retain their Gaussian multi-seed verification; and the simulation claim now covers both Gaussian and Poisson processes. The WISDM real-data variance claim remains explicitly not attempted.

**Reproduction bundle** (the experiment scripts) is attached as the artifact cell below.

**Rerun:**

    python kmarl_repro.py
    python sweep_thresh.py
    python audit_finite_sample_bounds.py --output evidence/finite_sample_bounds.json
    python poisson_reproduction.py --output evidence/poisson_figure13.json

numpy only, CPU, no external data.


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_d521b182f199", "created_at": "2026-07-20T08:33:23+00:00", "title": "Reproduction bundle", "artifact": "repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis/repro-bundle:v1", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis/repro-bundle:v1` · dataset

https://huggingface.co/buckets/kondratevakate/repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-surviva-artifacts#repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis/repro-bundle:v1
