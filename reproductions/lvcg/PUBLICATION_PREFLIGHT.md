# Publication preflight

## Result

The LVCG audit is locally complete and prepared for publication at:

`https://huggingface.co/spaces/kondratevakate/repro-lvcg-release-audit`

The publication attempt on `2026-07-30` was rejected before repository
creation by Hugging Face's 20-Spaces-per-day account limit (`HTTP 429`). An
authenticated read-only listing then confirmed that the target Space, trace
Dataset, and artifact Bucket do not exist. The local package is canonical
until remote creation succeeds.

## Passed checks

- official paper, source, repository commit, and artifact absence are hashed
  or pinned;
- six independent pytest checks pass;
- the paper equations have an independent deterministic geometry check;
- all released split CSVs are inventoried without reading any waveform;
- six pytest checks and the static Trackio logbook validator pass;
- PDF pages 6, 7, 15, and 16 were rendered and visually checked;
- no credentialed data, patient identifiers, or ECG signals are emitted;
- no leaderboard or third-party verdict was used.

## Publish command after quota reset

From `lvcg/`:

```powershell
trackio logbook publish kondratevakate/repro-lvcg-release-audit
```

After publication, verify HTTP 200, runtime status, remote revision, and the
remote `logbook.json` checksum before marking publication complete.
