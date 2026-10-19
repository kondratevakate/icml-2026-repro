# Publication preflight

## Result

Published:

- Space: `https://huggingface.co/spaces/kondratevakate/repro-caml-release-audit`
- static logbook:
  `https://kondratevakate-repro-caml-release-audit.static.hf.space/`
- Hugging Face revision:
  `dfa395f922530fa35dec4c1271b1565fe9216498`

Both URLs return HTTP 200, the Space reports `RUNNING`, and the static page
contains the CAML title. No trace Dataset or artifact Bucket was created
because the logbook contains neither traces nor embedded artifacts. The remote
`logbook.json` differs from the local file only by setting the unused
`workspace.bucket_id` to `null`.

## Passed checks

- paper, source, and official repository identities are pinned;
- the theorem counterexample and method checks are deterministic;
- six pytest checks pass;
- the released entrypoints compile;
- the mini Heat run is clearly labeled as a smoke test;
- PDF pages 4, 7, 9, 14, and 29 were rendered and visually checked;
- official caches and the isolated environment are ignored;
- no leaderboard or third-party verdict was used.

The publication command was:

```powershell
trackio logbook publish kondratevakate/repro-caml-release-audit
```
