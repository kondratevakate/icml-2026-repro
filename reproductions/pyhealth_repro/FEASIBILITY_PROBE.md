# PyHealth 2.0 Feasibility Probe

Date: 2026-07-29.

## Decision

`GO` for a source-level C1 verification and C4 falsification. `HOLD` C2 and C3
until the full MIMIC-IV v2.2 download and a persistent GCP machine are ready.

## Provenance

- arXiv source: `2601.16414v2`, updated 2026-05-17.
- arXiv source archive SHA-256:
  `8f592e99599e609ced3216700c1d4e878baffcb792c35a63a9c767c8224d078c`.
- official PyHealth tag: `v2.0.1`.
- official commit:
  `ed562121b5bae185b36322c64ce6215c2095dd50`.

## Findings

- The paper source contains explicit category tables for datasets, tasks,
  models, and interpretation methods. C1 can be audited without protected data.
- The anchored C4 challenge claim does not match Table 2. Table 2 reports
  mortality code counts `34/27/51`, not `7/24/51`.
- WSL Ubuntu 24.04 and Python 3.12 are available. A prior smoke run completed,
  but the old Windows log is incomplete and is not reused as evidence.
- The paper's MIMIC-IV benchmarks use v2.2 and an AMD EPYC 7513 workstation
  with 1 TB RAM. Demo data cannot establish C2 or C3.

## Stop conditions

- Stop C1 if any claimed threshold or named witness cannot be tied to both the
  paper inventory and the pinned release.
- Stop C4 if the seven-line statement and Table 2 cannot be cleanly separated.
- Do not start C2/C3 locally merely to produce a toy verdict.
