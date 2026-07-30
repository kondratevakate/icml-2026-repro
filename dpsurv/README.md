# DPsurv release audit

Independent reproduction package for:

**DPsurv: Dual-Prototype Evidential Fusion for Uncertainty-Aware and
Interpretable Whole-Slide Image Survival Prediction** (`RKqL4GYXz3`, arXiv
`2510.00053`).

Prepared result: **4/12** against six prespecified two-point claims, compared
with a frozen **5/12** forecast.

The package audits all released TCGA folds, executes the dual-prototype
evidential model and loss on deterministic synthetic inputs, checks GRFN
belief/plausibility bounds and lambda semantics, and reproduces the empty
component and visualization entrypoint failures. It does not substitute
synthetic results for the unreleased TCGA empirical artifacts.

Run:

```powershell
.\.venv\Scripts\python.exe audit_dpsurv.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe repro_dpsurv/build_logbook.py
.\.venv\Scripts\python.exe ..\validate_icml_logbook.py
```

See `FROZEN_FORECAST.md`, `INDEPENDENT_CHECKS.md`, and
`PUBLICATION_PREFLIGHT.md` for the evidence boundary.
