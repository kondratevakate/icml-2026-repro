# xMAE release audit

Run from this directory:

```bash
python audit_artifacts.py
python -m pytest -q
```

The audit uses only the frozen official repository artifacts. It does not
treat the made-up checkpoint or synthetic PVC data as evidence for the
paper-level downstream, cohort, timing, or generalization claims.
