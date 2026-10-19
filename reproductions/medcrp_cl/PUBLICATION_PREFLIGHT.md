# Publication preflight

## Status

The local package and static Trackio logbook are prepared and validated. Remote
publication is deferred because the same Hugging Face account has an
already-confirmed limit of 20 new Spaces per day. The earlier DC-PnPDP attempt
returned HTTP 429 with an approximately 24-hour retry instruction; no redundant
creation request is made while that cap remains active.

Intended Space:

`kondratevakate/repro-medcrp-cl-release-audit`

Intended command:

```powershell
trackio logbook publish kondratevakate/repro-medcrp-cl-release-audit
```

## Passed

- frozen forecast written before checkpoint loading or released-code execution;
- arXiv PDF/source, GitHub commit, and Hugging Face model revision pinned;
- modality and EWC checkpoint hashes match their immutable Hub LFS digests;
- nine deterministic CPU tests pass;
- all seven released Python files compile;
- paper pages 5-8 rendered and visually checked;
- official partition, Welford statistics, LoRA isolation, and EWC state audited;
- headline and robustness claims separated from release-level checks;
- local logbook validator passes;
- official artifacts and rendered pages are excluded from Git;
- no leaderboard, third-party verdict, or prior logbook used.

## Canonical blockers

- exact processed train/validation/test splits are not released;
- no run-level metric outputs, predictions, seeds, or baseline artifacts;
- checkpoint records CRP alpha 2.0 while paper/code specify 5.0;
- `requirements.txt` is not pip-installable;
- full rerun requires multiple external medical datasets, CLIPSeg weights, and
  substantial GPU training;
- Appendix Proposition 1's zero-error conclusion does not follow for
  overlapping fixed-variance Gaussians.

