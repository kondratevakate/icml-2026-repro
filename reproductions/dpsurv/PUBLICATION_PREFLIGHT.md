# Publication preflight

## Status

The local package and static Trackio logbook are prepared for validation.
Remote publication is deferred because the Hugging Face account has an
already-confirmed limit of 20 new Spaces per day. A prior publication attempt
returned HTTP 429 with an approximately 24-hour retry instruction, so no
redundant creation request is issued while the cap remains active.

Intended Space:

`kondratevakate/repro-dpsurv-release-audit`

Intended command:

```powershell
trackio logbook publish kondratevakate/repro-dpsurv-release-audit
```

## Passed

- forecast frozen before dependency installation or official-code execution;
- arXiv PDF/source and official GitHub commit pinned and hashed;
- all 25 five-cohort outer folds checked for case and slide leakage;
- 12 deterministic CPU tests pass;
- all 30 released Python files compile;
- paper method, result, ablation, and proof pages visually checked;
- synthetic dual-prototype forward/loss/backward independently executed;
- empirical claims separated from mechanism-level checks;
- official artifacts, virtual environment, and rendered pages excluded from Git;
- no leaderboard, third-party verdict, or prior logbook used.

## Canonical blockers

- WSI, UNI2-h features, PANTHER embeddings/prototypes, checkpoints, predictions,
  summaries, and run logs are not released;
- 86 unique cohort rows have no DSS endpoint and are filtered by the trainer;
- non-default lambda training scales both terms by lambda instead of using
  complementary weights;
- evaluation reverses the paper's Bel/Pl lambda direction;
- an all-low-prevalence component produces zero prototypes and crashes;
- the visualization encoder entrypoint is shadowed by a broken second
  definition;
- empirical ablation, robustness, clinical-coherence, and runtime artifacts are
  absent.
