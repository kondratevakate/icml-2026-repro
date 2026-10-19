# PyHealth 2.0 Source-Level Reproduction

Paper: **PyHealth 2.0: A Comprehensive Open-Source Toolkit for Accessible
and Reproducible Clinical Deep Learning** (`gMLVFN9hl8`, arXiv
`2601.16414v2`).

This bundle independently audits three challenge claims against the arXiv
source and the official PyHealth `v2.0.1` release. It does not use protected
clinical data.

## Frozen inputs

- Paper source: `https://export.arxiv.org/e-print/2601.16414v2`
- Official repository: `https://github.com/sunlabuiuc/PyHealth.git`
- Official tag: `v2.0.1`
- Official commit: `ed562121b5bae185b36322c64ce6215c2095dd50`

The source archive and official checkout are kept one directory above this
bundle during local execution. Their relevant file hashes are recorded in the
evidence JSON.

## Run

Only Python's standard library is required:

```bash
python prepare_sources.py
python audit_claims.py
python -m unittest discover -s tests -v
python validate_evidence.py
```

## Verdicts

| Claim | Verdict | Finding |
| --- | --- | --- |
| C1 toolkit coverage | `VERIFIED` | Paper tables enumerate 22 datasets, 43 tasks, 28 models, and 7 attribution methods. The pinned release exports more than every claimed threshold and contains witnesses for EHR, imaging, signals, genomics, Attention-Grad, GIM, DeepLift, and SHAP. |
| C2 mortality throughput | `NOT ATTEMPTED` | Requires full MIMIC-IV v2.2 and the exact multi-baseline worker sweep. |
| C3 drug/LOS throughput | `NOT ATTEMPTED` | Requires full MIMIC-IV v2.2 and the exact multi-baseline worker sweep. |
| C4 seven-line/Table 2 claim | `FALSIFIED` | The paper separately says training can take seven lines, but Table 2 reports mortality code counts of 34 for PyHealth 2.0, 27 for PyHealth 1.16, and 51 for Pandas. The anchored challenge claim incorrectly attributes 7/24/51 to Table 2. |
| C5 community/tutorial/R claim | `INCONCLUSIVE` | The release has 50+ example/tutorial files and the paper links RHealth. No immutable roster or measurement procedure supports an independent historical audit of "400+ members." |

The C4 verdict is narrow: it falsifies the anchored composite claim, not the
separate statement that a particular training path can be expressed in seven
lines.
