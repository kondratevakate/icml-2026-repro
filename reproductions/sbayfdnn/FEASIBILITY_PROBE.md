# Feasibility probe: sBayFDNN

Paper: *Sparse Bayesian Deep Functional Learning with Structured Region
Selection*.

OpenReview: `3IFIedDIoN`. arXiv: `2602.20651`.

Probe date: 2026-07-29.

## Primary artifacts

- Official code:
  `https://github.com/mengyunwu2020/sBayFDNN`
- Pinned code revision:
  `276d87949c8b973bf8b6748aa8df4f56086e063e`
- arXiv source SHA-256:
  `f9dae2ec3769e93f5887eca672e0d94c04fcf41b50580c0dc37a070ddeef2ccf`

No leaderboard, peer reproduction, or third-party verdict was inspected or
used as evidence.

## Artifact findings

The official repository contains a CPU-compatible synthetic demo, synthetic
data generators, the first-layer spike-and-slab optimizer, plug-in posterior
inclusion probabilities, Laplace evidence, and interval metrics. The notebook
contains a completed author run, but cached author output is provenance
context only and is not accepted as independent execution evidence.

The release does not contain the ECG or Tecator preparation/evaluation
pipelines used for the two real-data claims. The paper links the public raw
datasets, but reconstructing undocumented preprocessing, splits, baselines,
and repeated fits is not a local throughput route.

## Theory findings

- Theorem 5.4 asserts existence in a network class with parameter bound `E_n`
  but imposes no lower bound on `E_n`. Set the strictly positive `E_n=1/n`,
  use width one, and choose constant `X`, constant nonzero `beta`, and the
  identity link. All stated assumptions hold, while every admissible network
  output is `O(1/n)`. Its uniform error tends to one as the claimed upper
  bound tends to zero.
- Theorem 5.7 states `epsilon_n^2 <= C * complexity_n`, while its proof needs
  `complexity_n <= C' * epsilon_n^2`. This is recorded as a proof gap rather
  than a falsification of the theorem conclusion.

## Decision

`GO` for a method, approximation, and theorem-contract audit.

Prepared forecast: **4/12**.

Stop at source/theory plus a compact independent synthetic mechanism run. Do
not claim the ECG or Tecator tables without their missing release pipelines.
