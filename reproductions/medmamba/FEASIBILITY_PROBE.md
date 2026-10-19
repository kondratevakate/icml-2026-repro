# MedMamba Feasibility Probe

Paper: *MedMamba: Multi-View State Space Models with Adaptive Graph Learning
for Medical Time Series Classification* (`qPqJH0heR0`, arXiv `2605.24961v1`).

## Provenance

- Author repository: `https://github.com/zhangda1018/MedMamba`
- Pinned author commit: `418da50664338bc1d766394ee9c231496ab4de97`
- arXiv source archive SHA-256:
  `5bbbe280ecb9583c43cb764037678d5f9759a6b3eb7654d6afb8b97b8ccb7793`
- arXiv PDF SHA-256:
  `806fa52910a1648216a25728ca15146095a8d723ba2c7a6639ee6f6355566e4f`

No leaderboard entries or peer reproductions were inspected.

## Released execution surface

The repository contains one runnable shell configuration, for APAVA. It
depends on datasets prepared by the external Medformer repository and on
`mamba-ssm` with CUDA. It contains no checkpoints, cached predictions,
released metric files, or scripts for ADFTD, PTB, PTB-XL, and TDBRAIN.

The released requirements also contain the malformed token
`nvidia-cuda-runtime-cu1212.1.105`. The test checkpoint loader hard-codes
`map_location='cuda'`. These defects do not prevent source-level or CPU
architecture audits, but they block a clean canonical local run.

## Frozen forecast

This forecast is frozen before executable audits, after provenance and static
source inspection:

- realistic: `8/12`;
- stretch: `10/12`;
- expected local time: 3-5 hours;
- empirical five-dataset result: `0/2` until fresh canonical data runs exist.

The local route is unusually high-throughput despite the training burden:
several central architecture claims can be tested without fitting a model.

## Decision

`GO` for an implementation-conformance reproduction.

Stop conditions:

- do not award an empirical-result claim from paper tables or author prose;
- do not treat a CPU Mamba substitute as evidence about Mamba accuracy;
- use the substitute only to expose shapes, data dependencies, invariances,
  and gradient connectivity in the released MedMamba glue code;
- keep all official code and paper source outside the publishable bundle.

