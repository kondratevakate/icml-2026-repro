# Protocol 3: what blocks verification of claims in audited ICML 2026 papers

Frozen 2026-10-17 before any claim was coded. Git commit provides provenance.
Companion to `verdict_audit/PROTOCOL.md` (corpus-level verdict analysis), which
found that medical claims are judged verified less often and inconclusive more
often. This protocol asks why, using Kate's own release audits.

## Unit and sample

Primary sample: the 16 audit folders that contain all four documents
`OFFICIAL_ARTIFACTS.md`, `FROZEN_FORECAST.md`, `CLAIM_DECOMPOSITION.md` and
`INDEPENDENT_CHECKS.md`:

3dmedagent, bayes_causal_meta, came_grad, caml, dc_pnpdp, dpsurv,
flow_density_ratio, glean, globalhealthatlas, lvcg, ma_rag, medcrp_cl,
ophiuchus, spmind, sprout, xmae.

Unit of analysis: one scored claim in `INDEPENDENT_CHECKS.md`. Papers are
clusters. Other audit folders (earlier formats) are excluded from primary
analysis; they may be cited as case studies only, labelled as such.

Sources a coder may read: the four documents above, `PUBLICATION_PREFLIGHT.md`
and `README.md` in the same folder. Nothing else (no verdict data, no public
logbooks, no other coder's file).

## Coders

Coder 1: this Claude session. Coder 2: an independent Claude subagent that
receives only this protocol and the folder paths, and is told not to open any
`inventory_*` file. Each writes its own CSV. Disagreements are resolved after
agreement is computed; both raw files and the resolved file are kept.

The audits themselves were produced with agent assistance under frozen
protocols; they are agentic release audits, not manual expert reviews.

## Codebook

Paper-level file `papers_coderN.csv`:

| Field | Values |
|---|---|
| folder | folder name |
| orid | OpenReview id if stated, else blank |
| healthcare | `yes` if the paper's main task or main evaluation data are clinical, biomedical or public-health; else `no` |
| forecast_score, forecast_max | from FROZEN_FORECAST |
| prepared_score, prepared_max | from INDEPENDENT_CHECKS |
| code_release | `full` (training and evaluation code), `partial`, `none` |

Claim-level file `claims_coderN.csv`:

| Field | Values |
|---|---|
| folder, claim_no | claim order as listed in INDEPENDENT_CHECKS |
| claim_type | `theory` (proof, identity, bound); `mechanism` (method executes or is implemented as described); `data_protocol` (dataset, splits, corpus construction); `headline_performance` (main reported metric values); `secondary_empirical` (ablation, robustness, calibration, efficiency, interpretability, human study) |
| score, max | points as given |
| outcome | `full` (score = max), `partial`, `none` (score 0) |
| blockers | every reason the audit gives for withheld points, `;`-separated, from: `raw_data_unavailable` (data restricted, DUA, not released); `derived_artifacts_missing` (features, embeddings, predictions, logs, metric outputs, generated texts or labels not released); `weights_missing` (checkpoints not released); `code_missing_or_broken` (relevant code absent or fails to run); `paper_code_divergence` (released code contradicts the paper's stated method or equation); `compute_or_scale` (needs resources beyond the audit); `underspecified_paper` (paper lacks details needed to reimplement); `other`. Empty if outcome is `full`. |
| primary_blocker | one blocker from the list by precedence: paper_code_divergence > code_missing_or_broken > raw_data_unavailable > weights_missing > derived_artifacts_missing > compute_or_scale > underspecified_paper > other; `none` if outcome is `full` |
| divergence | `yes` if the audit documents a contradiction between released code and the paper for this claim, else `no` |
| pointer | at most 25 words locating the evidence in the audit text |

## Prespecified analyses

Run on healthcare = yes papers; healthcare = no papers reported alongside as
a within-sample comparison, not as a population estimate.

* **Q1 (primary).** Among claims with outcome not `full`, the share whose
  blockers include `derived_artifacts_missing` or `weights_missing` in papers
  with `code_release` = `full` or `partial`. Hypothesis: code is released but
  the intermediate artifacts needed to check headline numbers are not.
  Reported with a paper-cluster bootstrap CI (2000 resamples, seed 20261017).
  The statement "code release did not make claims checkable" is used only if
  this share is at least 0.5 and the CI lower bound exceeds 0.3.
* **Q2.** Distribution of primary blockers by claim type, especially
  `headline_performance` versus `theory`/`mechanism`.
* **Q3.** Number of papers with at least one documented paper-code divergence,
  each listed with its pointer.
* **Q4.** Forecast calibration: prepared minus forecast score per paper, mean
  and range; share of papers whose prepared score falls within the forecast's
  stated plausible range if one is given. Descriptive only (n = 16).
* **Reliability.** Cohen's kappa between coders for `healthcare`,
  `claim_type`, `outcome` and `primary_blocker`. The blocker taxonomy is
  reported as reliable only if kappa for `primary_blocker` is at least 0.60;
  otherwise results are presented as single-coder descriptive coding.
  Agreement on `score` must be exact (copied numbers); any mismatch is a data
  extraction error and is corrected before analysis.

## Evidence boundary

* Scores are Kate's prepared scores from her own audits, not the challenge
  judge's verdicts and not authors' confirmation.
* Sixteen papers selected by Kate's interest and feasibility: not a random
  sample of ICML or of medical ML.
* A documented divergence is an audit finding; it becomes a confirmed error
  only after author response or independent replication.
* Aggregate counts and per-paper case descriptions only; no participant other
  than Kate is named.
