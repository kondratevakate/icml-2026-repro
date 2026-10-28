# Tailored scoring rules feasibility probe

Paper: *Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An
Application to Causal Inference* (`JTwryHNicJ`, arXiv `2606.03332v1`).

## Provenance

- Primary paper: `https://arxiv.org/abs/2606.03332`
- OpenReview: `https://openreview.net/forum?id=JTwryHNicJ`
- arXiv source archive SHA-256:
  `7862c54d7da9cabecfefbc7bd2308c7cdff507223be68ed9c9ebaada73fe1129`
- arXiv PDF SHA-256:
  `adc0defaed5b133c8128b1ee3cb0962e07e6ab828700817855a3a6876f4fccaa`

The source contains the full theorem, proofs, scoring-rule derivation,
canonical-link derivation, experimental protocol, and result tables. No public
author code repository is linked in the paper or was found by a direct
title/author search.

No leaderboard entry or peer reproduction is used for target selection,
forecasting, or evidence.

## Released execution surface

The central mathematical claims are fully specified and can be audited with
SymPy, NumPy, and exact algebra on CPU:

- the IPW MSE upper bound and its bias/variance decomposition;
- the task-divergence curvature;
- the closed-form partial losses and strict propriety;
- the quartic canonical probability mapping and canonical gradient;
- the fully specified synthetic Kang-Schafer data-generating process.

The broad empirical comparison cannot be exactly rerun from the paper alone:
the per-run datasets, splits, seeds, tuning grids, and implementation are not
released. Paper tables and plots are not reproduction evidence.

## Frozen forecast

Frozen before symbolic audits, mutation tests, or simulation:

- realistic: `8/12`;
- stretch: `10/12`;
- strict floor: `6/12`;
- local effort: 5-8 hours;
- GPU: not required for the mathematical and synthetic claims.

## Decision

`GO` for a CPU mathematical and synthetic reproduction.

Stop conditions:

- distinguish local curvature matching from a global optimality guarantee;
- do not infer benchmark replication from values printed in tables;
- require a mutation or exhaustive numerical check for each algebraic claim;
- treat Kang-Schafer as a scoped synthetic mechanism check, not a reproduction
  of all reported experiments;
- keep IHDP, Jobs, and ACIC conclusions inconclusive without released runs.

