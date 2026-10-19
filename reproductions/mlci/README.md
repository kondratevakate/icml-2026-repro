# A Machine-Learned Comorbidity Index

Paper: arXiv 2606.17450, OpenReview `C6ZTjSXbz7`.

This directory prepares an independent reproduction while the full MIMIC-III
and MIMIC-IV tables are downloading. The current bundle audits the method
construction in C1, the rank-one threshold theory in C5, and both cohort
construction paths on synthetic integration fixtures. It does not claim the
MIMIC result tables.

Run:

```bash
cd repro_mlci
python run_all.py
```

See `FEASIBILITY_PROBE.md` and `CLAIM_DECOMPOSITION.md` before assigning any
empirical verdict. After the download finishes, follow `DATA_RUNBOOK.md`.
