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

## Medical reproducibility project page

Open the interactive map as a GitHub Pages project page:
[`kondratevakate.github.io/icml-2026-repro/`](https://kondratevakate.github.io/icml-2026-repro/).
The source file is [`docs/medical-reproducibility.html`](docs/medical-reproducibility.html);
[`docs/index.html`](docs/index.html) redirects the project-page root to it.

The page is a public audit of how reproducible medical and life-science ICML
papers look from the challenge metadata one month after the conference. It
starts with the full scale of the corpus, then narrows to the medically relevant
papers, and finally focuses the heatmap on papers that combine synthetic
experiments with real or open datasets.

Headline counts shown on the page:

- `6,341` total ICML papers in the challenge metadata.
- `703` life-science or medical papers, `11.1%` of the full corpus.
- `355` synthetic-plus-real/open-data target papers used for the detailed
  reproducibility heatmap.

The focused heatmap answers the practical reproduction questions: whether the
paper was reproduced in the contest, how many claims were checked, whether
authors released code, whether weights/checkpoints are available, and whether
GPU use appears necessary. Restricted/private and unknown data are shown in the
medical data-type carpet above the heatmap instead of being hidden inside the
open-data subset. Dataset size is intentionally excluded until a dedicated
artifact-size audit is run.

The generated audit tables are in [`audit/`](audit/), and the reproducible
pipeline is [`audit_medical_repro.py`](audit_medical_repro.py). To publish the
page on GitHub, enable Pages for this repository from the `docs/` folder on the
default branch.

## Reproducing locally

Each paper's folder is self-contained: read the script's docstring for the transcribed paper spec, then
run it directly (Python 3.11, numpy + torch CPU, no scipy/sklearn — see HANDOFF.md's environment
section). Downloaded datasets and vendored external code are not included in this repo; HANDOFF.md
documents where to get them.
