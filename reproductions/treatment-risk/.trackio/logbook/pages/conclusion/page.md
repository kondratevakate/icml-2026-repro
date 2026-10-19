# Conclusion

The high-throughput local scope succeeds. Lemma 4.1 and the Section 5.1
synthetic behavior are independently reproduced, including exact equality
checks and destructive controls that fail when the sensitivity model is
invalid. Theorem 4.3 has a real proof defect at Eq. (18), but the practical
finite-grid algorithm can be certified by a different first-unsafe argument;
this is recorded as partial rather than forced into a binary verdict.

The defensible score target is `4/10`, with possible partial credit on C2. STAR
and IST remain explicitly unattempted until the exact third-party cohorts are
available. No patient data, GPU, author code, or precomputed author outputs were
used.

## Reproduction commands

```powershell
python audit_lemma41.py --output evidence/lemma41_finite_audit.json
python audit_theorem43.py --maximum-grid-size 10 --output evidence/theorem43_audit.json
python reproduce_synthetic.py --runs 1000
```
