# ICML 2026 agent reproduction challenge: reproductions and verdict audit

This repository has two parts.

1. **Reproductions.** Independent reproductions of ICML 2026 papers submitted to the
   [ICML 2026 agent reproduction challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)
   under [kondratevakate](https://huggingface.co/kondratevakate). Each published logbook lives on its own
   Hugging Face Space (tagged `icml2026-repro` and `paper-<orid>`); this repository holds the scripts,
   logbook mirrors and reproduction bundles behind them.
2. **Verdict audit.** Meta-research on the challenge itself, from its public verdicts: how verdicts from
   many teams were aggregated, and why medical and life-science papers were rarely decided. Frozen
   protocols, analysis code, coding sheets, results and run logs.

## Contents

| Folder | What is in it | Start with |
|---|---|---|
| [`reproductions/`](reproductions/) | 47 folders, one per reproduced paper: scripts, `.trackio/` logbook mirror, `repro_*` bundle | [`reproductions/INDEX.md`](reproductions/INDEX.md): paper, Space and the judge's verdict counts per folder |
| [`verdict_audit/`](verdict_audit/) | Protocols 1–9 with code, coding sheets, results and the drafts of two PSE9 abstracts | [`verdict_audit/README.md`](verdict_audit/README.md), then [`verdict_audit/STUDIES.md`](verdict_audit/STUDIES.md) |
| [`notes/`](notes/) | Working notes: lessons learned and judged scores, target selection, model and subscription baselines | [`notes/HANDOFF.md`](notes/HANDOFF.md) |
| [`tools/`](tools/) | Logbook scaffold and validator, medical target scan, the project-page builder with its tests and the check behind it, the index builder | the docstring of each script |
| [`audit/`](audit/), [`docs/`](docs/) | Audit tables and the source of the medical reproducibility map | [the map section](#medical-reproducibility-map) and the [live page](https://kondratevakate.github.io/icml-2026-repro/medical-reproducibility.html) |

## Papers built on this repository

| Paper | Status | Draft | Evidence |
|---|---|---|---|
| Voters: how the ICML 2026 agent reproduction challenge aggregated verdicts | PSE9 2027 abstract, submitted 2026-10-23 | [`verdict_audit/voters/PAPER_DRAFT.md`](verdict_audit/voters/PAPER_DRAFT.md); submitted text in [`verdict_audit/PSE9_SUBMITTED.md`](verdict_audit/PSE9_SUBMITTED.md) | [`verdict_audit/voters/`](verdict_audit/voters/README.md) |
| Well documented, rarely picked: medical AI papers under agentic reproduction (Protocol 9) | PSE9 2027 abstract, submitted 2026-10-23; full paper in preparation for medRxiv, developed further in a separate repository | submitted text in [`verdict_audit/PSE9_SUBMITTED.md`](verdict_audit/PSE9_SUBMITTED.md) | [`verdict_audit/readiness/README.md`](verdict_audit/readiness/README.md) |
| Unfalsifiable by default: medical papers in the ICML 2026 agent reproduction challenge (Protocols 6–8) | earlier draft, superseded by the row above | [`verdict_audit/readiness/ABSTRACT_DRAFT.md`](verdict_audit/readiness/ABSTRACT_DRAFT.md) | provenance table in its Appendix A |

## Reproducing the verdict audit

```bash
python -m pip install -r verdict_audit/requirements.txt    # Python 3.11
python verdict_audit/reproduce.py --list                   # the analysis steps
python verdict_audit/reproduce.py                          # rerun them and compare with the committed results
```

`reproduce.py` reruns each analysis from the committed coding sheets and the downloaded inputs, compares every
JSON result with the committed one, restores the committed files, and writes a log with the environment, the
git commit and a per-step status to [`verdict_audit/logs/`](verdict_audit/logs/). The last full run
(2026-10-19, 41 minutes on CPU) matched the committed results in 15 of 15 steps. The inputs that are not in
the repository (the challenge's verdict snapshot, logbook pages, conference paper lists, full texts) and the
scripts that fetch them are listed in [`verdict_audit/README.md`](verdict_audit/README.md). Coding by
language-model subagents is not rerun: its prompts and outputs are committed and the analyses start from them.

## Reproducing a paper reproduction

Each folder under `reproductions/` is self-contained: read its README or the script's docstring (the paper's
specification as transcribed), then run the script. Most use Python 3.11 with numpy and CPU torch; folders
with their own `requirements.txt` pin their environment. Downloaded datasets, the papers' official code and
paper texts are not included; [`notes/HANDOFF.md`](notes/HANDOFF.md) and, where present, a folder's
`OFFICIAL_ARTIFACTS.md` say where to get them.

## Medical reproducibility map

Live page: [`kondratevakate.github.io/icml-2026-repro/medical-reproducibility.html`](https://kondratevakate.github.io/icml-2026-repro/medical-reproducibility.html)
(GitHub Pages, served from `docs/` of this branch). It is one self-contained file,
[`docs/medical-reproducibility.html`](docs/medical-reproducibility.html), which also works when downloaded and
opened locally: static SVG, no script, no embedded data blob.

The page opens on the corpus and then presents the Protocol 9 result, the second PSE9 abstract
*Well documented, rarely picked*, as five steps, each a figure with its confidence intervals.

0. **What was read.** One block per venue, block area proportional to the full texts coded there: 3,278 in
   all, `icml` 987, `neurips` 918, `iclr` 774, `cvpr` 215, `iccv` 207, `eccv` 177. Both the count and its
   life-science / other split are printed inside the block. The split is not encoded as area, because the
   other group is an equal matched sample drawn per venue-year and so carries no information.
1. **On paper.** Rate differences, life-science minus other, over the nine venue-years: code released `+13.1`
   points, main data openly downloadable `+15.0`, weights released `+28.6`, variance reported `+24.4`. The
   protocol pre-registered the opposite direction for L9.1-L9.3 and none of those labels was assigned.
   Variance did not meet the agreement gate, so coder 2 coded it for every paper and every disagreement was
   adjudicated before the numbers were computed.
   A second figure gives the same four axes as levels rather than differences, with the counts behind them:
   code `59.4%` (874 of 1,471) against `46.2%` (747 of 1,618), main data open `83.7%` (725 of 866) against
   `67.3%` (422 of 627), weights `48.6%` (136 of 280) against `19.4%` (96 of 495), variance `66.4%`
   (842 of 1,269) against `42.1%` (578 of 1,372). These are the shares from the final codes the analysis
   uses, so the difference printed beside each axis is the crude one (`+13.2`, `+16.4`, `+29.2`, `+24.2`)
   and not the stratified estimate above it. `unclear` leaves the denominator rather than counting as a
   negative. (Until 2026-10-30 the levels were drawn from coder 1's raw sheet, which showed variance at
   `75.1%` against `50.7%` before adjudication.)
2. **In practice.** Among papers attempted in the challenge, `46.4%` of the 110 medical ones reached a
   verified or falsified verdict against `65.5%` of the 1,821 others: a gap of `-19.1` points, permutation
   p = 0.0005. A second figure repeats the question on the matched set, one square per attempted paper, over
   the 656 coded ICML 2026 papers only: 51 of 110 medical settled against 71 of 112 others from the same
   venue, `-17.0` points, CI `[-29.7, -4.4]`. The two gaps differ because they have different comparison
   groups - every other attempted paper in the challenge, against other papers from the one venue - and both
   are on the page so neither reading has to stand in for the other.
3. **Where the gap goes.** The same gap after adjusting for the first team's settle rate on its other papers
   (`-13.0`), then for that and the claims' check cost (`-5.6`, CI `[-11.9, +1.4]`).
4. **The two factors.** Teams that took a medical paper first settled `40.0%` of their other papers against
   `60.3%`; `56.7%` of medical claims needed retraining or a large-scale run against `39.1%`, from two blind
   coders on 1,083 claims.
5. **The caveat on any artifact map.** Agreement between what a paper states and what a reproducer obtained:
   kappa `0.22` for a code link against the authors' code actually being run, `0.13` for stated data access
   against data obtained, `0.55` for weights.

A provenance table at the foot of the page lists every number with the result file and key it was read from.
Colour is used on one figure only, the verdict squares, where it names a verdict; its settled squares are
grouped first so the comparison is the length of two blocks and does not depend on telling two hues apart.
Everything else is monochrome.

The page is generated, not hand-written. [`tools/build_medical_map.py`](tools/build_medical_map.py) reads the
frozen Protocol 9 results and emits the HTML, so the page can be diffed against the JSON and regenerated:

```bash
python tools/build_medical_map.py                      # regenerate the page
python -m pytest tools/test_build_medical_map.py       # check it
```

[`tools/test_build_medical_map.py`](tools/test_build_medical_map.py) checks what a reader cannot: that the
counts behind each figure come out of the coding sheets with `unclear` left in the denominator rather than
counted as a negative, that block areas track their paper counts, that no label runs past the box it sits in,
and that two builds of the page are byte-identical.

### The layer this page used to show

Until this build the page showed a different figure: 703 ICML 2026 papers selected from the challenge's area
tags and a title scan, with artifact signals from the challenge metadata and a July 2026 leaderboard snapshot.
Checked against the two-blind-coder label for the same venue, which marks 328 papers life-science,
[`tools/check_map_against_labels.py`](tools/check_map_against_labels.py) found that set agrees at kappa `0.59`:
386 of its papers are not life-science and 11 life-science papers are missing. Its artifact signals agree with
the coded reference at kappa `0.19` for code and `0.10` for restricted data. The check is in
[`audit/map_label_check.json`](audit/map_label_check.json).

That layer is retired. `docs/assets/medical-reproducibility-map.png` is a static render of it, kept so the
check above can be read against the picture it describes; it is not a render of the current page. The audit
tables behind it stay in [`audit/`](audit/), produced by
[`tools/audit_medical_repro.py`](tools/audit_medical_repro.py).

## How this was made

Reproductions and analyses were written with AI coding agents, directed and reviewed by the author; commits
made with Claude carry a `Co-Authored-By` trailer. In the verdict audit every protocol freeze
and bulk download was approved by the author before it happened, and blind coding was done by Claude
subagents whose prompts are committed. The author took part in the challenge; the verdict audit excludes the
author's own logbooks.

## Licence and citation

Code is MIT-licensed ([`LICENSE`](LICENSE)). Text, protocols, coding sheets and aggregate results are CC BY 4.0
([`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0)). Third-party material keeps its own terms and is not redistributed
here: the papers and their official code, the challenge's datasets, and other teams' logbooks.

To cite this repository, use [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button reads it).
