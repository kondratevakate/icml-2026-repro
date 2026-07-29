# Publication preflight

- [x] Forecast frozen before independent checks.
- [x] Primary paper, arXiv source, official repository commit, and released CSV
  frozen with hashes.
- [x] No leaderboard or third-party reproduction verdict used as evidence.
- [x] Paper pages 3, 4, 7, and 8 visually inspected after PDF rendering.
- [x] Independent CSV audit generated from raw released rows.
- [x] Four unit/mutation tests pass.
- [x] Official repository passes `python -m compileall -q .`.
- [x] Official CLI help runs.
- [x] CPU dry run exercises CSV selection and canonical T1S cap without claiming
  missing visual/model execution.
- [x] Claims C3–C6 remain explicitly inconclusive.
- [x] Local Trackio logbook validates.
- [x] Public Space publishes and remote validation passes.

Final score before publication: **4/12**.

Published Space:
`https://huggingface.co/spaces/kondratevakate/repro-3dmedagent-artifact-audit`

Postflight:

- Space revision:
  `e35c12b1c7f6655f5914f8af02c6ab29f0377e57`
- runtime stage: `RUNNING`
- repository page, static root, and remote `logbook.json`: HTTP 200
- remote ICML logbook validation: passed
