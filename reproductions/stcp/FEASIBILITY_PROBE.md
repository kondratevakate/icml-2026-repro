# StCP Feasibility Probe

Date: 2026-07-29.

## Decision

`GO` for a local theorem, mutation, synthetic, and medical-artifact
reproduction of *Stable Localized Conformal Prediction via Transduction*.

## Frozen forecast

- Realistic: `8/12`.
- Stretch: `10/12` after a fresh medical DermaMNIST or TissueMNIST run.
- Expected local wall time: `4-7 hours`.
- Stop condition: do not claim a medical empirical result from bundled tables
  or checkpoints alone; require a fresh run over released data.

This forecast was recorded before auditing the proofs or running the official
experiment code.

## Medical relevance

The method is evaluated on DermaMNIST and TissueMNIST. The paper constructs a
high-risk target group with more balanced rare disease/tissue classes and
frames the task as tissue classification for medical risk stratification.

## Provenance

- OpenReview forum ID: `lSMTccAN61`.
- arXiv: `2605.01452v1`.
- arXiv source archive SHA-256:
  `f188ddb7acec78014f2ebc185c01496e2c3e99bb2197f6ff598b0d1cc4e376de`.
- arXiv PDF SHA-256:
  `5e8b8b72e3841069b0aad01769d4ac6559af0eb62654058223dd1101efdb6518`.
- Author repository:
  `https://github.com/OswinMin/SLCP`.
- Pinned author commit:
  `84118ddf19efe211250a5f024c1be5a3c8f47b4b`.

## Available artifacts

- Full StCP/SLCP method implementation.
- Reproducible synthetic DGPs: Quad, Softplus, and LogAbs.
- Canonical `n,m` grids and 50-repeat scripts.
- DermaMNIST data and trained Derma/Tissue feature checkpoints.
- Regression datasets and experiment scripts.
- Full arXiv source, algorithms, theorem statements, proofs, and tables.

## Initial runnable path

The documented simulation entrypoint is:

```text
python SimuAnalysis/run_shared.py logabs 30 500
```

The full run uses 50 repeats, 500 test points, neural conditional-distribution
estimation, GLCP and CQR variants, and multiple lambda values. A reduced
fixed-seed smoke path is acceptable only for feasibility; claim verification
requires either the full canonical run or a mathematically equivalent
independent implementation with repeated calibration samples.

## Risks

- The repository has no pinned dependency file.
- The paper source and repository summary tables are not byte-identical,
  indicating that cached text is not sufficient reproduction evidence.
- Initial inventory did not surface TissueMNIST; the later recursive audit
  found both its raw `.npz` file and checkpoint.
- Real experiments can be materially slower than the theorem and synthetic
  audits.

## Probe outcome

The pinned author code completed an end-to-end LogAbs smoke run on Windows
CPU at `n=30`. A deliberately smaller `n=10` probe fails in the released
selector because its upper quantile level exceeds one; this does not affect
the paper's `n=30` configuration.

Both `dermamnist.npz` and `tissuemnist.npz` are present in the pinned author
repository. The earlier TissueMNIST availability risk above is therefore
resolved. Fresh multi-repeat medical execution remains pending because the
real-data procedure is substantially heavier than the smoke path.

The independent audit prepared `6/12` before fresh canonical experiments.
The original `8/12` forecast remains achievable only if the 50-repeat LogAbs
run receives an empirical verdict.
