# GLEAN independent release audit

This bundle audits the official paper/source and the two source datasets
without using leaderboard entries or third-party verdicts. It never prints,
copies, or packages clinical rows.

## Run

From this directory:

```powershell
python audit_glean.py `
  --clinical-root ..\..\data\mimic-iv-ext-cdm-1.1 `
  --notes-root ..\..\data\mimic-iv-note-2.2\release\note
pytest -q
```

The audit writes only aggregate counts, schemas, checksums, and mathematical
comparisons to `evidence/audit.json`.

## Evidence boundary

The credentialed datasets are repository-ignored and excluded from the
published Space. The author did not release GLEAN code, trajectories,
predictions, calibration rows, active-verification traces, or clinician
annotations. Consequently, the package supports 2/12 points and explicitly
marks the remaining claims unsupported rather than reconstructing them from
paper tables alone.

