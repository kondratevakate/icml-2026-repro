# CAME-Grad independent equation and release audit

This CPU-only deterministic bundle audits *The Double Dilemma in Multi-Task
Radiology Report Generation: A Gradient Dynamics Analysis and Solution*
(OpenReview `T7y2wavrFM`, arXiv `2605.22635v2`).

## Run

```bash
pip install -r requirements.txt
python audit_claims.py
python -m unittest discover -s . -p "test_*.py" -v
```

For a primary-artifact audit, place the official repository under
`../official/code` and the extracted TeX at
`../official/arxiv-source/example_paper.tex`.

## Outcome

- C1: **VERIFIED AT EQUATION LEVEL WITH QUALIFICATIONS (2/2)**.
- C2: **NOT ESTABLISHED BY PRINTED EQUATIONS (0/2)**.
- C3–C5: **INCONCLUSIVE, NOT RERUN (0/6)**.
- Prepared score: **2/10**.

The independent implementation follows only equations 8–14. It is not the
withheld official optimizer. It verifies trust-region, magnitude and fusion
identities, exposes Eq. 10's zero-denominator case, and demonstrates that equal
update norms need not imply equal diffusion covariance.

Published table arithmetic is checked but never promoted to experimental
evidence. No medical images, reports, author weights, leaderboards, or
third-party verdicts are included.
