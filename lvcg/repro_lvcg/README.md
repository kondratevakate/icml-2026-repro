# LVCG independent release audit

This package audits the official paper source and repository for *Learning
Cardiac Latent Representations in Vectorcardiogram Space* without using a
leaderboard or third-party verdict.

Run from this directory:

```powershell
python audit_lvcg.py
python -m pytest -q
```

The audit contains no clinical waveforms or patient-level output. It checks
artifact identity, package importability, fixed-geometry equations, released
split integrity, table arithmetic, and paper/code conformance.
