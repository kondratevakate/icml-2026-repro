# SPROUT release audit

Independent reproduction package for:

**Supervise Less, See More: Training-free Nuclear Instance Segmentation with
Prototype-Guided Prompting** (`qYfNhYenuu`, arXiv `2511.19953`).

Prepared result: **5/12** against six prespecified two-point claims, compared
with a frozen **4/12** forecast.

The package audits the official self-reference mask, progressive partial-OT,
containment-aware soft-NMS, and evaluator on deterministic synthetic inputs. It
also tests the POT proof boundary and records the release artifacts needed—but
not provided—for the pathology benchmark and robustness tables.

Run:

```powershell
.\.venv\Scripts\python.exe audit_sprout.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe repro_sprout/build_logbook.py
.\.venv\Scripts\python.exe ..\validate_icml_logbook.py
```

See `FROZEN_FORECAST.md`, `INDEPENDENT_CHECKS.md`, and
`PUBLICATION_PREFLIGHT.md` for the evidence boundary.
