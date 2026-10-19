# Publication preflight

## Result

The GLEAN audit is locally complete and prepared for publication at:

`https://huggingface.co/spaces/kondratevakate/repro-glean-release-audit`

The first publication attempt on `2026-07-30` was rejected before repository
creation by Hugging Face's 20-Spaces-per-day account limit (`HTTP 429`). A
read-only authenticated listing confirmed that neither the Space nor its trace
Dataset or artifact Bucket exists yet. The local package remains the canonical
publication source.

## Passed checks

- paper PDF and arXiv source hashes match the frozen inventory;
- all 12 MIMIC-IV-Ext CDM payload hashes match the official checksum manifest;
- credentialed clinical data remain under the repository-wide ignored `data/`;
- `official/` paper/source caches are ignored;
- the aggregate audit emits no patient identifiers or clinical text;
- four independent pytest checks pass;
- static logbook generation succeeds;
- `validate_icml_logbook.py` passes for
  `kondratevakate/repro-glean-release-audit`;
- key PDF pages 6, 7, 8, 13, 14, and 16 were rendered and visually checked;
- no leaderboard or third-party verdict was used.

## Publish command after quota reset

From `glean/`:

```powershell
trackio logbook publish kondratevakate/repro-glean-release-audit
```

After publishing, verify HTTP 200, Space runtime status, remote revision, and
the remote `logbook.json` checksum before marking publication complete.

