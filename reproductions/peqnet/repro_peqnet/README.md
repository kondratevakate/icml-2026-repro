# PEQ-Net independent reproduction bundle

This CPU-only bundle audits six claims from arXiv `2605.14284v2`.

## Run

```bash
python audit_claims.py --output evidence/claims_audit.json
python -m unittest discover -s tests -v
```

When the pinned arXiv source is available locally, add:

```bash
python audit_claims.py \
  --source-root ../official/source \
  --output evidence/claims_audit.json
```

The numerical audit is deterministic. It independently implements the
Gaussian-kernel MMD and metric-MDS policy representation, checks the paper's
printed DGP for executable singularities, and constructs a counterexample to
the LTMLE Lipschitz step used in Theorem 4.2.

## Scope

- C1-C3 are `INCONCLUSIVE`: no author code, protected MIMIC-III input, and an
  undefined printed lag coefficient prevent a faithful 20-seed reproduction.
- C4 is `VERIFIED` with an explicit qualification about MMD versus squared MMD
  and approximate versus exact MDS preservation.
- C5 is `FALSIFIED AS A LIPSCHITZ GUARANTEE`: two proof steps do not follow
  from the theorem assumptions.
- C6 is `INCONCLUSIVE` until the complete MIMIC-IV cohort path exists.

No patient-level data, official paper source, or official figures are included
in the publishable bundle.
