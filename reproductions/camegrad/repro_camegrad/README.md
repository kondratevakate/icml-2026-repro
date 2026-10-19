# CAME-Grad mathematical reproduction

This CPU-only bundle audits five claims from *The Double Dilemma in Multi-Task
Radiology Report Generation: A Gradient Dynamics Analysis and Solution*
(arXiv `2605.22635v2`, OpenReview `T7y2wavrFM`).

## Result

- C1 `VERIFIED`: the exact two-task interaction identity holds in 1,000
  deterministic trials and exact opposing gradients collapse to zero energy.
- C2 `FALSIFIED AS AN UNCONDITIONAL GUARANTEE`: the declared Stage 1 problem
  has a feasible exact solution whose inner product with one task is negative.
- C3 `PARTIALLY VERIFIED`: the conditional scalar covariance identity holds,
  but epsilon means Stage 2 does not enforce the stated target norm exactly.
  The stochastic covariance of the full mechanism remains untested.
- C4 `QUALIFIED`: a fixed example executes the exact Stage 1 dual solution,
  Stage 2 normalization, and legal Stage 3 fusion. Stage 1 improves both tasks,
  while the final direction harms one task. This limits the broad narrative;
  the paper formally defines geometric validity for the Stage 1 direction and
  acknowledges conflict resurgence at high fusion coefficients.
- C5 `INCONCLUSIVE`: the clinical averages were not rerun. The public
  repository imports an absent `CAME_Grad.py`, and its README says the core
  optimizer and model weights are temporarily withheld.

Prepared score: `6/10`.

## Run

```bash
python prepare_official.py
python audit_claims.py \
  --official-code-root ../official/code \
  --paper-source ../official/source/example_paper.tex \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
```

The audit uses NumPy only and is deterministic under seed `20260729`.

## Integrity

The publishable bundle contains no author code, paper source, model weights,
medical images, reports, or patient records. The evidence records the pinned
author commit and SHA-256 hashes of the audited files.

No leaderboard entries or peer reproductions are used as evidence.
