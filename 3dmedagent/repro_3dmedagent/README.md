# 3DMedAgent independent artifact audit

This package audits two reproducible claims from the official 3DMedAgent
release:

1. the released OAMI → CFLT → T1S architecture and its declared five-turn
   paper configuration;
2. the composition and basic integrity of the released DeepChestVQA CSV.

Run:

```powershell
python repro_3dmedagent/audit_artifacts.py
python -m unittest repro_3dmedagent/test_audit_artifacts.py
```

The audit uses only Python's standard library. It does not rerun clinical model
comparisons: the official compact release lacks the complete volumes, derived
artifacts, credentials, canonical predictions, and expert traces needed for
those claims.
