# sBayFDNN reproduction

This CPU bundle independently audits six claims from *Sparse Bayesian Deep
Functional Learning with Structured Region Selection* (OpenReview
`3IFIedDIoN`, arXiv `2602.20651`).

Prepared score: **4/12**.

## Results

- C1 `VERIFIED` (2/2): the released plug-in PIP agrees with an independent
  group-normal Bayes derivation, and the spline-to-region mapping agrees with
  an independent knot-support implementation.
- C2 `FALSIFIED AS STATED` (2/2): Theorem 5.4 does not constrain its network
  parameter bound `E_n`. With strictly positive `E_n=1/n`, width one,
  constant `X` and `beta`, and identity link, every stated assumption holds
  but every admissible output is `O(1/n)`; its error tends to one while the
  claimed bound tends to zero.
- C3 `INCONCLUSIVE PROOF GAP` (0/2): Theorem 5.7 states
  `epsilon^2 <= complexity`, while its proof requires
  `complexity <= constant * epsilon^2`. This does not alone falsify the
  contraction conclusion.
- C4 `INCONCLUSIVE` (0/2): the proof adds an unstated posterior-to-MAP
  equivalence premise.
- C5/C6 `NOT EXECUTED` (0/4): the release omits the ECG and Tecator pipelines.

## Run

Prepare and verify the pinned author artifacts, then run:

```bash
python prepare_official.py
python audit_claims.py \
  --code-root ../official/code \
  --paper-source ../official/paper/ICML-FDNN.tex \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
```

The audit uses no cached notebook metrics as evidence and does not need a GPU
or medical records.

## Integrity

The publishable bundle contains no author repository, paper source, raw ECG,
credentials, leaderboard data, or peer-reproduction evidence.

Primary sources: [arXiv](https://arxiv.org/abs/2602.20651),
[OpenReview](https://openreview.net/forum?id=3IFIedDIoN), and
[official code](https://github.com/mengyunwu2020/sBayFDNN).
