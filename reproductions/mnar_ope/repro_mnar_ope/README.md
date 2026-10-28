# MNAR OPE Independent Claim Audit

Paper: **Off-Policy Evaluation for Missingness-Aware Policies in MDPs with
Rewards Missing Not at Random** (`vpSFJoxyDz`, arXiv `2606.20206v1`).

This bundle audits the five challenge claims against a pinned paper source and
official repository. It uses an analytic counterexample and independent CPU
simulation; it does not require MIMIC data or a GPU.

Published logbook:
`https://huggingface.co/spaces/kondratevakate/repro-off-policy-evaluation-for-missingness-aware-policies-in-mdps-with-rewards-missing-not-at-r`.

## Frozen inputs

- Paper source: `https://export.arxiv.org/e-print/2606.20206`
- Official repository: `https://github.com/NAIVlab/ShadOPE`
- Official commit: `4231ba5d46046c66c0efae6d58662bfca5087147`

The source tree and repository checkout are kept under `../official/` during
local execution.

## Run

```bash
python audit_claims.py
python -m unittest discover -s tests -v
```

The audit writes `evidence/claims_audit.json`.

## Current verdict policy

| Claim | Prepared verdict | Evidence path |
| --- | --- | --- |
| C1 bridge existence | `FALSIFIED AS WRITTEN` | Wrapped-Gaussian inverse problem satisfies the listed assumptions but has no square-integrable bridge. |
| C2 relevance | `VERIFIED` | Exact source equation, official DGP dependency, and residual association on the observed subset. |
| C3 bridge error | `VERIFIED` | Exact theorem, algebraic reduction, and finite-dimensional inequality test. |
| C4 policy-value rate | `VERIFIED` | Proof ledger, recurrence unrolling, concentrability product, and critical-radius substitution. |
| C5 no future dependence | `VERIFIED` | Exact source equation, generator dependency audit, and nested logistic diagnostic. |

The C1 verdict is intentionally narrow. The paper states "and some regularity
conditions" and supplies Picard conditions in its appendix. The anchored
challenge claim omits those conditions.
