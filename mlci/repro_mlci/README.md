# Independent MLCI audits

Current scope:

- C1: Appendix-width DeepSets encoder and weighted multi-outcome nHSIC
  construction.
- C5: rank-one reduction and monotone threshold theorem.
- Cohort construction: schema, filtering, labels, patient-disjoint split, and
  ICD normalization for MIMIC-III and MIMIC-IV.

Run:

```bash
python run_all.py
```

Dependencies: Python 3.11, NumPy 1.26+, and PyTorch 2.0+ CPU.

These scripts use synthetic diagnostic inputs to audit the method and theorem.
They do not reproduce C2-C4 or C6 and are not a substitute for MIMIC runs.

Check whether the downloaded raw tables are ready:

```bash
python inspect_mimic_tables.py --root C:\Projects\data\digitaltwin
```

The paper does not disclose the deterministic split hash or salt. The cohort
builder therefore records both choices and reports differences from every
published cohort and split count. See `../DATA_RUNBOOK.md`.
