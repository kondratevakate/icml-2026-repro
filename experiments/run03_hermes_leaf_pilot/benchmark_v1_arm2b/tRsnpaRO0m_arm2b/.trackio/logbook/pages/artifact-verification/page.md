## Artifact verification

`check_reproducibility.py` re-runs all six scripts and asserts, per claim, (a) the stored
`results/claim<N>.json` is **not stale** (mtime newer than its script) and (b) a re-run is
**byte-identical by SHA-256** — one assertion that subsumes exit-0, schema validity, verdict validity
and seed determinism. Output in `repro_check.json`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Artifact verification"}\n-->
