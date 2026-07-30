# CAML independent release audit

This package audits the official source and code for *Mitigating Gradient
Pathology in PINNs through Aligned Constraint*. It uses no leaderboard or
third-party verdict.

Run in the isolated environment from this directory:

```powershell
..\.venv\Scripts\python.exe audit_caml.py --include-mini-heat
..\.venv\Scripts\python.exe -m pytest -q
```

The mini Heat run is a runtime smoke test only. It is never presented as a
reproduction of the five-seed GPU table.
