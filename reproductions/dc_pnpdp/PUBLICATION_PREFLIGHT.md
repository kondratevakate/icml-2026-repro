# Publication preflight

## Status

Local package prepared and validated. Publication was attempted on 2026-07-30
but Hugging Face rejected creation with HTTP 429 because the account had
reached the limit of 20 new Spaces per day. The response advised retrying in
about 24 hours.

Authenticated follow-up checks confirmed that no partial remote resources
were created:

- Space `kondratevakate/repro-dc-pnpdp-release-audit`: absent;
- Dataset `kondratevakate/repro-dc-pnpdp-release-audit-traces`: absent;
- Bucket `kondratevakate/repro-dc-pnpdp-release-audit-artifacts`: absent.

The intended publication command is:

```powershell
trackio logbook publish kondratevakate/repro-dc-pnpdp-release-audit
```

## Passed

- frozen forecast written before code execution;
- paper, source, and official repository identities pinned;
- six deterministic pytest checks pass;
- all 22 released Python files compile;
- key PDF pages 7, 8, 16, 17, 22, 23, and 24 rendered and visually checked;
- canonical GPU/data-dependent claims explicitly separated from local checks;
- official artifacts and caches ignored by Git;
- no leaderboard or third-party verdict used.

## Canonical blockers

- CPU-only local environment;
- CUDA-only `torch-radon` CT operator;
- external AbdomenCT-1K volumes and multi-gigabyte diffusion checkpoint;
- no released MRI implementation;
- no cached reconstructions, metrics, ablations, or timings.
