# Publication preflight

## Status

Local package prepared and validated. Remote publication is blocked by the
Hugging Face account's already-confirmed limit of 20 new Spaces per day. The
immediately preceding DC-PnPDP publication attempt returned HTTP 429 and
instructed retrying in about 24 hours, so this package does not issue a
redundant creation request while that daily cap is active.

Intended Space:

`kondratevakate/repro-bayes-causal-meta-release-audit`

Intended command:

```powershell
trackio logbook publish kondratevakate/repro-bayes-causal-meta-release-audit
```

## Passed

- frozen forecast written before code execution;
- paper, source, and official repository identities pinned;
- six deterministic pytest checks pass;
- all 16 released Python files compile;
- adaptive training and BALD expert-inference smoke paths exercised;
- PDF pages 6, 7, 8, 11, 12, and 13 rendered and visually checked;
- canonical UK Biobank claims separated from synthetic checks;
- official artifacts, environment, and caches ignored by Git;
- no leaderboard or third-party verdict used.

## Canonical blockers

- restricted UK Biobank cohort under project permission 77565;
- no clinical outputs or checkpoints;
- no cached 30-run synthetic results;
- incomplete training RNG seeding;
- paper/release prior-map mismatch.

