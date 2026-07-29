# Claim 4: seven-line Table 2 attribution

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pyhealth_c4_001", "created_at": "2026-07-29T14:30:00+00:00", "title": "Claim 4: seven-line Table 2 attribution"}
-->
**Anchored claim (verbatim).** "PyHealth 2.0 reduces the code required to
implement standard ML tasks (e.g., mortality prediction) to as few as 7 lines,
down from 24 lines in PyHealth 1.16 and up to 51 lines with raw pandas (Table
2)."

**Verdict -- FALSIFIED AS ANCHORED.**

The audit machine-parses the table labelled `tab:loc_comparison` in the public
arXiv v2 source. Its mortality-prediction column is:

| Method | Table 2 mortality LOC |
| --- | ---: |
| Pandas | 51 |
| PyHealth 1.16 | 27 |
| MEDS ETL + MEDS_Reader | 43 |
| PyHealth 2.0 | 34 |

Therefore the challenge claim's Table 2 attribution `7/24/51` disagrees with
the paper for both PyHealth rows. The paper does separately state in the
abstract and methodology that a model-training path can take seven lines. That
separate statement does not change the values in Table 2, and Table 2 does not
report 24 lines for PyHealth 1.16.

The falsification is deliberately narrow: the anchored composite claim
conflates a standalone seven-line statement with Table 2. It does not establish
that no possible PyHealth workflow can be written in seven lines.

**Mutation control.** Replacing the two Table 2 entries with 7 and 24 makes all
five audit checks pass and flips the verdict to `VERIFIED`.
