# Verdict audit: what the ICML 2026 agent reproduction challenge could decide

This folder holds the meta-research built on the public verdicts of the
[ICML 2026 agent reproduction challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
Two abstracts for PSE9 (Perspectives on Scientific Error 9, Copenhagen 2027)
come from it:

1. **Voters study** — how the challenge aggregated verdicts from many teams:
   [`voters/PAPER_DRAFT.md`](voters/PAPER_DRAFT.md), with a draft note to the
   organisers ([`voters/NOTE_TO_ORGANISERS_DRAFT.md`](voters/NOTE_TO_ORGANISERS_DRAFT.md), not sent).
2. **Unfalsifiable by default: medical papers in the ICML 2026 agent
   reproduction challenge** — [`readiness/ABSTRACT_DRAFT.md`](readiness/ABSTRACT_DRAFT.md).
   Every number in it is traced to a result file in its Appendix A.

## How the studies were run

* Each study starts from a protocol frozen in git before its data were
  downloaded or analysed. Amendments and notes are dated and state what had
  been seen when they were written; post hoc analyses are labelled as such.
* Prespecified labels are reported whether or not they were met. Several
  were not (see the protocols and the drafts).
* Result files are JSON, written by the analysis scripts, next to a manifest
  with input SHA-256 hashes, code hashes, seeds and the git commit of the run.
* Commit history is part of the record: freezes, data draws and results are
  separate commits.

## Map

| Folder | Protocol | Frozen | Main result files | Used in |
|---|---|---|---|---|
| this folder (`PROTOCOL.md`, `analyze.py`, `confirm.py`) | Reproducibility verdicts in the challenge: discovery run and confirmation run | 2026-10-17 | `results/results.json`, `results_confirmation/results.json` | background |
| [`healthcare/`](healthcare/) | What blocks verification of claims in audited ICML 2026 papers (the author's release audits) | 2026-10-17 | `healthcare/results.json` | abstract 2 |
| [`e1_adjusted/`](e1_adjusted/) | Protocol 4: medical label check (E0) and the adjusted medical verdict gap (E1) | 2026-10-17 | `e1_adjusted/results_e1_validated.json` | abstract 2 |
| [`falsifiability/`](falsifiability/) | Protocol 6: how decidable medical papers were under crowd reproduction | 2026-10-18 | `falsifiability/results6.json` | abstract 2 |
| [`voters/`](voters/) | Voters protocols 1–3: aggregation of verdicts; first logbooks and exposure to published verdicts; resubmission (paused) | 2026-10-18 | `voters/results/*.json` | abstract 1 |
| [`readiness/`](readiness/) | Protocol 7: readiness of medical papers across ML venues; Protocol 8: hidden science; Protocol 9: 2025–2026 baseline of reproducibility practice (in progress) | 2026-10-18 / 2026-10-19 | `readiness/results/readiness.json`, `p8.json`, `p8_posthoc.json`, `p9_frame.json` | abstract 2 |

Protocol numbers were assigned per study as the work grew, so two folders
have a "Protocol 3"; refer to protocols by folder.

## Data that are not in this repository

Only aggregates, coding sheets keyed by public ids, and code are committed.
Third-party content (paper texts, other teams' logbooks) is not redistributed.

| Input | Source | Where the scripts expect it |
|---|---|---|
| Challenge verdicts | Hugging Face dataset `ICML-2026-agent-repro/verdicts`, revision `f0c5a8b369d6d618ecc9fa060aa2e2f27f9e2602`, file `verdicts.json` (SHA-256 `652f1c16…`) | `../data/verdicts_confirmation_2026-10-17/verdicts.json` (repository root) |
| Logbook pages, commit histories, file trees of public Spaces | Hugging Face Spaces at the judged revision, via `voters/download*.py` | `voters/data/` (git-ignored); aggregate digests in `voters/results/` |
| Conference paper lists (ICLR, NeurIPS, ICML, CVPR, ICCV, ECCV 2024–2026) | Conference virtual sites, via `readiness/download_sources.py` and `readiness/fetch_abstracts.py` | `readiness/data/` (git-ignored); hashes in `readiness/results/sources_manifest.json` |
| Full texts, protocol 8 | arXiv HTML or PDF, via `readiness/fetch_p8_fulltext.py` | `readiness/data/p8/` (git-ignored) |
| Full texts, protocol 9 | arXiv, PMLR and CVF open access, via `readiness/fetch_p9.py` | `DATA_ROOT` (default `C:\Projects\data`), `datasets/literature/ml-venue-fulltext/2025-2026-v1/` |

## Rerunning

* Environment: Python 3.11 and [`requirements.txt`](requirements.txt) (pinned; `voters/requirements.txt` pins
  the voters study alone).
* One command reruns every analysis and checks it against the committed results:
  `python reproduce.py` (`--list` shows the steps). It sets the committed outputs aside, reruns each script,
  compares the new JSON with the committed JSON, puts the committed files back, and writes
  `logs/reproduce_<UTC time>.json` with the environment, the git commit and a status per step. Keys that are
  known to differ for a documented reason (a machine-dependent counter, a provenance field added by hand) are
  listed in `reproduce.py` and reported in the log instead of being compared.
* Last full run: [`logs/reproduce_20261019T074627Z.json`](logs/reproduce_20261019T074627Z.json), 15 of 15
  steps match, 41 minutes on CPU. It ran the analysis code of commit `0d87130`, with `reproduce.py` and this
  documentation not yet committed.
* Language-model coding is not rerun: prompts and codes are committed and the analyses start from them.
* Run individual scripts from their folder without `python -I`; they import local modules. The study cards
  ([`STUDIES.md`](STUDIES.md) and each folder's README) give the order.
* Tests: `python -m pytest` in this folder, in `voters/`, in `e1_adjusted/` and in `readiness/`.

## Who did what

* Verdicts are the challenge's own, produced by its LLM judge
  (`zai-org/GLM-5.2`).
* Blind coding was done by Claude subagents; each protocol names the model
  and the prompts are committed next to the codes.
* Code, protocols and drafts were written in Claude Code (`claude-opus-5-5`),
  directed and reviewed by the author, who approved every protocol freeze and
  bulk download. Commits carry a `Co-Authored-By` trailer.
* Disclosure: the author took part in the challenge (namespace
  `kondratevakate`). Challenge analyses exclude the author's logbooks; the
  release audits in `healthcare/` are the author's own.

## Licence

Code: MIT ([`../LICENSE`](../LICENSE)). Protocols, drafts, coding sheets and
aggregate results: CC BY 4.0 ([`../LICENSE-CC-BY-4.0`](../LICENSE-CC-BY-4.0)).
Challenge data, paper texts and other teams' logbooks keep their own terms
and are not included.
