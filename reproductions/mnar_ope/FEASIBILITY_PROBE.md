# MNAR OPE Feasibility Probe

Date: 2026-07-29.

## Decision

`GO` for a local source, theorem, and CPU-simulation reproduction. The MIMIC-III
experiment is not needed for the five challenge claims and is excluded from the
first submission.

## Frozen forecast

- Realistic: `6/10`.
- Stretch: `8/10`.
- Expected wall time: `4-7 hours`.
- Stop condition: move to `HOLD` if fewer than three claims have evidence that
  is independent of merely restating the paper.

This forecast was recorded before auditing the proofs or running the
independent checks.

## Provenance

- OpenReview forum ID: `vpSFJoxyDz`.
- arXiv source: `2606.20206v1`, dated 2026-06-18.
- arXiv source archive SHA-256:
  `2a3ddc551f79f95d6e9a4448c61656c5b2dbcf828faa4a99662e67ef2345229c`.
- arXiv PDF SHA-256:
  `c0ee4b2218bed0ca6d7647d49b6cf56ed178eea01204e5779655c1c957957059`.
- Official repository: `https://github.com/NAIVlab/ShadOPE`.
- Official commit:
  `4231ba5d46046c66c0efae6d58662bfca5087147`.

## Live duplicate check

No Hugging Face Space matching paper ID `vpSFJoxyDz` was returned by the live
Hub API search on 2026-07-29. The local verdict snapshot from 2026-07-28 also
contains no judged attempt for this paper.

## Runnable path

The official CPU smoke path completed on Windows with Python 3.11:

```text
python -m scripts.simulation \
  --dataset ../../evidence/smoke_dataset.npz \
  --n 32 --T 2 --gamma 1.0 --seed 42 --device cpu
```

All five estimators ran and produced finite values. The official repository
also includes raw 50-seed simulation tables and precomputed MIMIC-III results.
Those cached outputs are useful provenance but are not treated as independent
reproduction evidence.

## Important claim-level finding

The anchored C1 claim lists Assumptions 2.1, 2.2, and 3.2-3.4 as sufficient for
bridge existence. The paper's theorem is more qualified: it adds "some
regularity conditions," and its appendix requires a Hilbert-Schmidt condition
and a Picard summability condition. The reproduction therefore tests the
anchored claim literally and separately preserves the paper's qualification.

