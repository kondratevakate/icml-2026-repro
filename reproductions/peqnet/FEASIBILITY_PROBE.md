# PEQ-Net Feasibility Probe

Date: 2026-07-29.

## Decision

`GO` for a local, CPU-only audit of the policy-embedding pipeline and Theorem
4.2. `HOLD` for the three semi-synthetic performance claims and the MIMIC-IV
case study.

## Frozen forecast

- Realistic: `4/12`.
- Stretch: `6/12` after the restricted-data path is restored.
- Expected local wall time: `2-4 hours`.
- Stop condition: do not independently rebuild the 500-epoch neural baselines
  unless author code or an exact processed cohort becomes available.

This forecast was recorded before implementing the numerical counterexamples.

## Provenance

- OpenReview forum ID: `bIcz7bIZSo`.
- arXiv: `2605.14284v2`, revised 2026-05-27.
- arXiv source archive SHA-256:
  `c060571a8da793a431c3136ca7133c99fad239134a11df8cd8d50870c3a7bd3f`.
- arXiv PDF SHA-256:
  `ca790971f328efa7122e0e322070abf4d91dae8f44bd695076f1e42275b50f00`.
- Public author repository: none found by exact-title, arXiv-ID, OpenReview-ID,
  and `PEQ-Net` searches on 2026-07-29.

## What is locally runnable

The paper source fully specifies the conceptual policy-embedding path:

1. Apply each policy to observed histories.
2. Compute pairwise Gaussian-kernel MMD values on history-action pairs.
3. Apply metric MDS.
4. Feed the per-step embeddings to a reverse-time RNN.
5. Condition a shared Q-function on the encoded policy tail.

An independent NumPy/scikit-learn implementation can exercise steps 1-3 and
audit the mathematical interface to steps 4-5 without protected data or a GPU.
The theorem and appendix are also available in full source form.

## What is blocked

### Semi-synthetic Tables 1 and 2

The experiments require 1,000 processed MIMIC-III trajectories, 20 seeds,
500-epoch DeepLTMLE/PEQ-Net training, and an unpublished DeepLTMLE
implementation. The released DGP is additionally undefined as written at
`i=1`, because it contains `(-1)^i / (1-i)` inside a sum beginning at one.

### MIMIC-IV sepsis case study

The local download does not yet contain complete MIMIC-IV `hosp/` and `icu/`
tables. The paper gives cohort-level inclusion criteria but no cohort SQL,
item-ID mapping, preprocessing code, model checkpoint, or random seeds.
Restricted patient rows must never be published to Git or a Space.

## Throughput conclusion

This is a viable `4/12` source-and-counterexample submission now. It is not a
responsible local target for the empirical claims until code or exact data
artifacts appear.
