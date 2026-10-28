# Voters study: aggregating verdicts in the ICML 2026 agent reproduction challenge

Protocols: `PROTOCOL.md` (Protocols 1-3, amendments, post hoc notes).
Paper draft (target: PSE9, Copenhagen 2027): `PAPER_DRAFT.md`. Draft note to organisers (not sent):
`NOTE_TO_ORGANISERS_DRAFT.md`.

## How this study was produced

* Code, protocols and drafts were written in Claude Code (desktop app) with
  the model `claude-opus-5-5`, directed and reviewed by the author in chat.
  The author approved each protocol freeze and each bulk download before it
  happened. Commits carry a `Co-Authored-By: Claude Opus 5.5` trailer.
* The Q3 blind coders were two Claude subagents running `claude-opus-5-5`
  (recorded in their transcripts). Their prompt is in
  `q3_codes/CODER_PROMPT.md`.
* `ACCESS_LOG.csv` lists the external access of the analysis session: every
  web fetch and web search, literature-database calls, Hugging Face API
  methods used in shell commands (counted) and subagent launches with their
  model. Built by `build_access_log.py` from the session transcript; command
  text, file contents and local paths are not included.

## Inputs and where they come from

| Input | Origin | Check |
|---|---|---|
| `data/verdicts_confirmation_2026-10-17/verdicts.json` | Dataset `ICML-2026-agent-repro/verdicts`, revision `f0c5a8b369d6d618ecc9fa060aa2e2f27f9e2602`, file `verdicts.json` | LFS SHA-256 equals the local file's (`652f1c16…`), checked 2026-10-18 |
| `verdicts_live.json`, `verdicts_latest_tmp.json` | Local snapshots saved 2026-07-22 and 2026-07-28 during the challenge | Hashes in `PROTOCOL.md`; not matched to a dataset revision |
| `data/spaces_2026-10-17/spaces.json` | Space ids and `createdAt`, collected on 2026-10-17 by an earlier session | Hash in `PROTOCOL.md`; **collection script not recorded** |
| `claims_anchored.json` | Anchored claim texts, local file dated 2026-07-18 | Hash in `PROTOCOL.md`; **origin not recorded in this folder** |
| Logbook pages, commit histories, file trees | Public Spaces at the judged revision (`sha`), via `download*.py` | Per-file hashes in `data/*_manifest.json` (git-ignored); aggregate digests (no per-Space entries) in `results/download_manifests_hashed.json` |

Page contents and Space ids are not committed (they identify participants).

## Environment

Python 3.11; packages in `requirements.txt`. Each analysis writes a
manifest with input, code and protocol hashes, the git commit, seed, resample
count and package versions.

## Run order

Network steps need a Hugging Face connection and are throttled under the API
rate limit. Analysis scripts refuse to overwrite existing result files, so a
re-run goes into a fresh clone.

| Step | Command | Output |
|---|---|---|
| 1 | `python download.py` | `data/pages/`, `data/pages_manifest.json` |
| 2 | `python analyze_voters.py` | `results/results.json`, `results/manifest.json` |
| 3 | `python q3_sheets.py`, then two blind coders with `q3_codes/CODER_PROMPT.md`, then `python q3_score.py` | `data/q3/`, `q3_codes/coder*_*.csv`, `results/q3.json` |
| 4 | `python posthoc_blog_logbooks.py` | `results/posthoc_blog_logbooks.json` |
| 5 | `python download_commits.py` | `data/commits_manifest.json` |
| 6 | `python analyze_exposure.py` | `results/exposure.json`, `results/exposure_manifest.json` |
| 7 | `python download_trees.py` | `data/trees_manifest.json` |
| 8 | `python review_checks.py` | `results/review_checks.json` |
| 9 | `python hash_manifests.py` | `results/download_manifests_hashed.json` |
| 10 | `python build_access_log.py <session.jsonl> <subagents dir>` | `ACCESS_LOG.csv` |
| tests | `python -m pytest -q` | unit tests for patterns and both analyses |

Protocol 3 analysis (`analyze_resubmission.py`) is written but not yet run.

## Claims in the paper and their evidence

| Paper claim | Script | Result key | Status |
|---|---|---|---|
| Agreement by category (falsified 0.300, verified 0.687; AC1 0.358) | `analyze_exposure.py` | `exposure.json: exploratory.protocol1_q2_prevalence_robust` | post hoc |
| kappa 0.202; collapsed 0.193; teams needed | `analyze_voters.py` | `results.json: Q2_effective_teams` | prespecified |
| Near-duplicates: 164 pairs, 17.7% of claims | `analyze_voters.py` | `results.json: Q1_copies` | prespecified |
| Majority 1/4, Dawid-Skene 0/4 on confirmed falsifications | `analyze_voters.py` | `results.json: Q4_aggregation` | prespecified (descriptive) |
| Cited logbooks were judged falsified (3 of 4) | `posthoc_blog_logbooks.py` | `posthoc_blog_logbooks.json` | post hoc |
| Dissenting falsifiers rated +0.60, win 35.5% | `analyze_exposure.py` | `exposure.json: H4.later_falsified` | prespecified (descriptive) |
| Exposure difference -0.011; permutation null -0.023 | `analyze_exposure.py` | `exposure.json: H1` | prespecified; bias note post hoc |
| Agent/model not recoverable | `q3_score.py` | `q3.json` | prespecified |
| Resubmission counts; latency tail; dissent spread; empty traces | `review_checks.py` | `review_checks.json` | post hoc |

Every number in `PAPER_DRAFT.md` is also listed in its Appendix A.

## Not done

* No clean re-run from scratch has been performed. Results are deterministic
  by construction (fixed seeds, stable hashes), but this has not been
  confirmed by re-execution.
