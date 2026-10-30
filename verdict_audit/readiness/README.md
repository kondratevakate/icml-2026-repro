# Readiness: protocols 7, 8 and 9

Three protocols in one file, [`PROTOCOL.md`](PROTOCOL.md), each frozen before its data were drawn:

| Protocol | Question | Frozen | Status |
|---|---|---|---|
| 7 | Do medical papers at ML venues (ICLR, NeurIPS, ICML, CVPR, ICCV, ECCV 2024–2026) declare code less often? | `3b9e5ea`; amendments `2d92340`, `f70d29e` | done; label not met (medical papers declare code more often) |
| 8 | Hidden science: what can a crowd check without the data? | `3217b19`; notes 8-1 to 8-4 | done; H8.1, H8.4, H8.5 not met |
| 9 | A 2025–2026 baseline of reproducibility practice in life-science ML, with a toy-data pilot | `839281e` | in progress |

The abstract built on them is [`ABSTRACT_DRAFT.md`](ABSTRACT_DRAFT.md); every number in it is traced to a key
in its Appendix A.

## Inputs (not committed)

* Conference paper lists, downloaded once by `download_sources.py` into `data/` (hashes in
  [`results/sources_manifest.json`](results/sources_manifest.json)); abstracts for the 2026 strata by `fetch_abstracts.py`
  and `fetch_iclr2026_abstracts.py`.
* The verdict snapshot `../../data/verdicts_confirmation_2026-10-17/verdicts.json` (protocols 7 R-A3 and 8).
* RECLAIM runs index (protocol 7 R-B1), recorded in `results/sources_manifest.json`.
* Protocol 8 full-text excerpts from arXiv (`fetch_p8_fulltext.py` into `data/p8/`).
* Protocol 9 full texts under `DATA_ROOT` (`fetch_p9.py`; default `C:\Projects\data`).

## Committed coding

* Protocol 7 medical labels: [`codes/labels.csv`](codes/labels.csv) (coder 1 on all candidates, coder 2 on
  25%, gate kappa 0.80; reached 0.96).
* Protocol 8: prompts in [`codes/p8/CODER_PROMPTS.md`](codes/p8/CODER_PROMPTS.md); codes and adjudications in
  [`codes/p8/`](codes/p8/).
* Protocol 9: frame [`results/p9_frame.json`](results/p9_frame.json); pilot prompts in
  [`codes/p9/PILOT_PROMPTS.md`](codes/p9/PILOT_PROMPTS.md); pilot inputs in [`pilot/`](pilot/); aggregate
  outcomes in [`results/p9_pilot_wave1.json`](results/p9_pilot_wave1.json) (Note 9-2).
* Protocol 9 release table: [`results/p9_papers.csv`](results/p9_papers.csv), one row per frame paper (4,340) with
  identifiers, title, group, full-text status and the final codes; no abstracts or full texts, which may not be
  redistributed. Built by `build_p9_release.py`; columns are in its docstring. `test_p9.py` checks that the
  protocol 9 prevalences recomputed from it match `results/p9.json`.

## Run

```bash
cd verdict_audit/readiness
# protocol 7
python download_sources.py              # once; keeps hashes stable, never re-downloads
python fetch_abstracts.py <venue-year> <host>
python build_candidates.py && python split_batches.py   # blind coding sheets
python score_labels.py                  # kappa gate -> results/labelling.json
python analyze_readiness.py             # R-A1, R-A2, R-A3, R-B1 -> results/readiness.json
# protocol 8
python build_p8.py && python fetch_p8_fulltext.py       # samples and excerpts
python analyze_p8.py score              # agreement and adjudication lists
python analyze_p8.py run                # H8.1-H8.5 -> results/p8.json
python posthoc_p8.py                    # Notes 8-2 and 8-3 -> results/p8_posthoc.json
# protocol 9
python build_p9_frame.py                # once -> results/p9_frame.json
python fetch_p9.py index && python fetch_p9.py run
python build_p9_release.py              # -> results/p9_papers.csv (publishable table)
python -m pytest test_readiness.py test_p9.py
```

Run the scripts from this folder without `python -I`: they import local modules.

## Outputs

`results/labelling.json`, `results/readiness.json` (+ manifest), `results/p8_coding.json`,
`results/p8.json` (+ `p8_manifest.json`), `results/p8_posthoc.json`, `results/p9_frame.json`
(+ `p9_frame_manifest.json`).

## Limits

One LLM judge; observational contrasts; language-model coders; excerpts can miss data statements; full
texts only where arXiv or open proceedings have them.
