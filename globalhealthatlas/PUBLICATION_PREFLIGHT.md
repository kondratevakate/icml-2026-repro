# GlobalHealthAtlas publication preflight

- [x] Candidate selected from the local medical/relevant shortlist without
  leaderboard or third-party verdict evidence.
- [x] arXiv v3 PDF and TeX source frozen and hashed.
- [x] Official GitHub revision and both linked Hub revisions frozen.
- [x] Hugging Face dataset availability checked through official Hub search
  and Dataset Viewer paths.
- [x] Forecast frozen before any numerical aggregation or code execution.
- [x] Six claim groups decomposed before final scoring.
- [x] Both main LoRA adapters downloaded and verified against Hub LFS SHA-256.
- [x] Detailed benchmark, incremental-SFT, transfer, robustness, and leakage
  exports independently recomputed.
- [x] Missing corpus, expert gold labels, stability matrices, raw predictions,
  prompts, perturbations, and seeds explicitly bounded.
- [x] Lightweight released code smoke-tested.
- [x] Four focused audit tests pass.
- [x] PDF pages 1, 6, 7, 8, 9, and 38 rendered and visually inspected.
- [x] Static Trackio logbook built, validated, published, and remotely checked.
- [x] Space repository, static root, and remote `logbook.json` return HTTP 200.
- [x] Remote/local `logbook.json` SHA-256 match:
  `3D5EA7354E7FC8095C54D75D5992C3E64F9EC83141DF98139B8AA1039479BA81`.

Prepared score: **6/12**.

Published Space:
`https://huggingface.co/spaces/kondratevakate/repro-globalhealthatlas-release-audit`

Published revision:
`90d52b86b178540b8af5ec8b5734028df3597fbf`.
