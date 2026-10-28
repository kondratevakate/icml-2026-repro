# Publication preflight

- [x] Primary paper, arXiv source, official repository, and commit frozen.
- [x] Forecast written before independent runtime and artifact checks.
- [x] No leaderboard or third-party verdict used.
- [x] Paper pages 3, 5, 6, 15, and 20 rendered and visually inspected.
- [x] Official checkpoint, HDF5, and metadata hashes recorded.
- [x] Strict checkpoint load and train/eval forward paths checked independently.
- [x] Synthetic HDF5/metadata alignment checked independently.
- [x] PVC random-encoder notebook path rerun independently.
- [x] Four focused tests pass.
- [x] Audit and test sources compile.
- [x] Protected-data/real-weight blockers are explicit.
- [x] C3-C6 remain inconclusive rather than inferred from paper tables.
- [x] Local Trackio logbook validates.
- [x] Public Space publishes and remote artifact postflight passes.

Prepared result before publication: **2/12**.

Published Space:
`https://huggingface.co/spaces/kondratevakate/repro-xmae-release-audit`

Postflight:

- Space revision:
  `1df2eeb92eacb3a0a660e74e8c15191f13b359b2`
- runtime stage: `RUNNING`
- repository page, static root, and remote `logbook.json`: HTTP 200
- remote `logbook.json` SHA-256 matches the local published artifact
- ICML logbook validation: passed
