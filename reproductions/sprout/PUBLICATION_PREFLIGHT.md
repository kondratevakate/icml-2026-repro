# Publication preflight

## Status

The local package and static Trackio logbook are prepared for validation.
Remote publication is deferred because the same Hugging Face account has an
already-confirmed limit of 20 new Spaces per day. A prior publication attempt
returned HTTP 429 with an approximately 24-hour retry instruction, so this
audit does not issue a redundant creation request while the cap remains active.

Intended Space:

`kondratevakate/repro-sprout-release-audit`

Intended command:

```powershell
trackio logbook publish kondratevakate/repro-sprout-release-audit
```

## Passed

- forecast frozen before dependency installation or official-code execution;
- arXiv PDF/source and official GitHub commit pinned and hashed;
- eight deterministic CPU tests pass;
- all 17 released Python files compile;
- paper method and proof pages rendered and visually checked;
- self-reference, partial-OT, refinement, and metric paths independently tested;
- empirical claims separated from mechanism-level checks;
- official artifacts, virtual environment, and rendered pages excluded from Git;
- no leaderboard, third-party verdict, or prior logbook used.

## Canonical blockers

- `OptimalTransport.solve()` reads an undefined `self.numItermax`;
- after the minimal runtime completion, plan mass is scaled by `N` relative to
  the paper formulation;
- convexity alone does not justify equality of particular optimal couplings;
- the documented preprocessed-dataset link is empty;
- the documented `docs/requirements.txt` path is misspelled in the release;
- SAM2 code/checkpoints are external and the default feature backbone does not
  resolve at the pinned freeze;
- no predictions, processed splits, metric outputs, robustness runs, or runtime
  traces are released.
