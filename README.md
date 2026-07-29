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
- `PAPERS.md` — target-selection reference material (own notes).
- Not included here (public on the challenge's own HF dataset, no need to duplicate):
  `papers.json`, `claims.json`, `claims_anchored.json`, `all_papers.csv`, `medical_neuro_papers.csv` —
  download from `https://huggingface.co/datasets/ICML-2026-agent-repro/challenge`.
  `claims_anchored.json` is the precise, judged version of the claim text — see HANDOFF.md for why the
  distinction between it and `claims.json` matters.

## Published logbooks

See the table in [`HANDOFF.md`](HANDOFF.md#опубликованные-логбуки-11-штук-реальные-баллы) for the
full list with real judged scores.

## Medical reproducibility map

An interactive public-facing audit prototype lives at
[`docs/medical-reproducibility.html`](docs/medical-reproducibility.html). For
GitHub Pages, enable Pages from the `docs/` folder and link to that page.

The map first shows the scale of medical/life-science papers inside all ICML
papers, then splits the medical set by data type, and then highlights the
simulation-plus-real-data subset. The heatmap below focuses on that subset and
shows whether each paper was reproduced in the contest, claim load, author code,
released weights/checkpoints, and whether GPU appears to be needed. The open
dataset row is omitted from that focused heatmap because the subset is open-data
by construction; restricted/private data remains visible in the medical
data-type strip above it. Dataset size is intentionally excluded until a
dedicated artifact-size audit is run.

## Reproducing locally

Each paper's folder is self-contained: read the script's docstring for the transcribed paper spec, then
run it directly (Python 3.11, numpy + torch CPU, no scipy/sklearn — see HANDOFF.md's environment
section). Downloaded datasets and vendored external code are not included in this repo; HANDOFF.md
documents where to get them.
