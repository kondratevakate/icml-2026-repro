# ICML 2026 Reproduction Challenge — working repo

Independent reproductions submitted to the [ICML 2026 Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)
under [kondratevakate](https://huggingface.co/kondratevakate). Each published logbook lives on its own
HuggingFace Space (tagged `icml2026-repro` + `paper-<orid>`); this repo holds the reproduction scripts,
working notes, and the full methodology writeup behind them.

**Start here:** [`HANDOFF.md`](HANDOFF.md) — target-selection criteria, every pitfall hit and what it
cost in points (real judged scores, not guesses), the current queue, and the working rules this project
follows.

## Layout

- One folder per paper (`calpro/`, `survfd/`, `kmarl/`, `mdrcp/`, `dcpnpdp/`, `confsleepnet/`,
  `solvable-ae/`, `uqct/`, `entropy/`, `ccd/`, `survival-eval/`, `proconmv/`) — each contains the
  verification script(s), a `.trackio/` logbook mirror, and a `repro_*` bundle staged for the published
  artifact.
- `PAPERS.md` / `all_papers.csv` / `medical_neuro_papers.csv` — target-selection reference material.
- `claims.json` / `claims_anchored.json` — the challenge's own claim text per paper (`claims_anchored.json`
  is the precise, judged version — see HANDOFF.md for why the distinction matters).

## Published logbooks

See the table in [`HANDOFF.md`](HANDOFF.md#опубликованные-логбуки-11-штук-реальные-баллы) for the
full list with real judged scores.

## Reproducing locally

Each paper's folder is self-contained: read the script's docstring for the transcribed paper spec, then
run it directly (Python 3.11, numpy + torch CPU, no scipy/sklearn — see HANDOFF.md's environment
section). Downloaded datasets and vendored external code are not included in this repo; HANDOFF.md
documents where to get them.
