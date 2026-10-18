# Protocol 7: reproduction readiness of medical papers across ML venues

Frozen 2026-10-18, approved by Kate in chat, before any venue file is
downloaded for analysis, any paper is labelled, and any readiness share is
computed. Git commit is the provenance. Part of the PSE9 abstract
"Unfalsifiable by default: medical papers in the ICML 2026 agent reproduction
challenge". Companion to protocols 3 (release audits), 4 (adjusted verdict
gap) and 6 (decidability) in `verdict_audit/`.

## Status of the evidence

Protocol 6 showed that medical ICML 2026 papers were attempted as often as
others but rarely decided, and protocol 4 that the judge was not harsher on
them within team and claim type. This protocol tests one mechanism outside the
challenge: are medical papers, at several venues, released in a form that a
third party can reproduce? Before freezing, only the structure of the sources
was inspected (field names, topic vocabularies, one record); no readiness
share was computed for any group.

## Sources (downloaded once, SHA-256 recorded in `results/manifest.json`)

* Conference virtual-site files `https://{venue}.cc/static/virtual/data/{venue}-{year}-orals-posters.json`
  for ICLR 2025, ICLR 2026, NeurIPS 2024, NeurIPS 2025, ICML 2025, ICML 2026.
* RECLAIM benchmark, dataset `Mithilss/reclaim`, revision
  `59099a8ececf954665dff90a41ad8541e3b21291`, file `eval_100.jsonl`; and the
  RECLAIM code repository `https://github.com/mithils3/reclaim` at the commit
  current on download (recorded).
* ICML 2026 challenge verdicts: `data/verdicts_confirmation_2026-10-17/verdicts.json`
  (dataset revision `f0c5a8b369d6d618ecc9fa060aa2e2f27f9e2602`, SHA-256
  `652f1c16…`), deduplicated as in protocol 1.
* Abstracts for RECLAIM papers from the arXiv API by the `custom_id` arXiv id.

Raw files go to `verdict_audit/readiness/data/` (git-ignored; they contain
author names). Only aggregates and per-paper labels keyed by OpenReview or
arXiv id are committed.

## Unit

One accepted paper per venue-year: items with `eventtype` Poster or Oral,
deduplicated by the OpenReview forum id in `paper_url` (else `sourceid`).
Items without an abstract are excluded and counted. Strata: the six
venue-years.

## Medical label

* Candidate if either holds:
  - `topic` is one of: `Applications->Health`, `Applications->Health / Medicine`,
    `Applications->Genetics, Cell Biology, Health, etc`,
    `Applications->Genetics, Cell Biology, etc`,
    `Applications->Neuroscience, Cognitive Science`,
    `Neuroscience, Cognitive Science`, `Computational Biology`,
    `Applications->Chemistry and Drug Discovery`;
  - title or abstract matches, case-insensitive, the regex
    `\b(patients?|clinical\w*|clinicians?|hospitals?|EHRs?|electronic health records?|medical|medicine|diseases?|diagnos\w*|prognos\w*|radiolog\w*|patholog\w*|histopatholog\w*|MRI|fMRI|CT scans?|X-rays?|ultrasound|ECG|EEG|surg\w*|cancers?|tumou?rs?|oncolog\w*|cardi\w*|alzheimer\w*|dementia|ICU|MIMIC\w*|sepsis|drugs?|pharmac\w*|therap\w*|health ?care|public health|epidemi\w*|biomedic\w*|proteins?|genes?|genom\w*|transcriptom\w*|single-cell|scRNA\w*|DNA|RNA|molecul\w*|enzymes?|antibod\w*|peptides?|biolog\w*|neurons?|neuroscien\w*|brain)\b`.
* Codebook: the E0 codebook of protocol 4 (`clinical`, `life_science`, `no`),
  from title and abstract only.
* Coder 1 (a Claude subagent) codes every candidate. Coder 2 (an independent
  Claude subagent) codes a random 25% of candidates (seed 20261018) and a
  random 200 non-candidates pooled over venue-years (seed 20261018), which
  coder 1 also codes. Coders see title and abstract only: no venue, topic,
  links, verdicts or each other's files. Their model is recorded from the
  subagent transcripts.
* Gate: Cohen's kappa (three categories) between coders on the candidate
  overlap at least 0.80. If met, coder 1's labels stand outside the overlap
  and overlap disagreements are adjudicated by this session with a written
  reason. If not met, coder 2 codes all remaining candidates and every
  disagreement is adjudicated; this is reported.
* Groups: **medical** = candidates with label `clinical` or `life_science`;
  **other** = non-candidates. Candidates labelled `no` are excluded from both
  groups and counted. Secondary groups: `clinical` only, `life_science` only.
  The 200 non-candidates give the share of medical papers the prefilter
  misses (consensus of both coders), with a Wilson interval.

## Outcomes (per paper)

* **O1 code link (primary):** any link, in the `url` field or in an
  `eventmedia` item of type `URL`, whose host is `github.com`, `gitlab.com`,
  `bitbucket.org`, `huggingface.co`, `codeberg.org`, `sourceforge.net`,
  `gitee.com` or `anonymous.4open.science` (the last counted separately too).
* **O2 any artifact link (sensitivity):** any such link with a host other than
  `openreview.net`, the venue's own domain, `youtube.com`, `slideslive.com`,
  `x.com` or `twitter.com`.
* **O3 restricted-data mention (descriptive):** title or abstract matches,
  case-insensitive,
  `\b(MIMIC\w*|eICU|UK Biobank|ADNI|PhysioNet|electronic health records?|EHRs?|in-house|proprietary|private (data|dataset|cohort)s?|institutional (data|dataset|cohort)s?|hospital (data|records|cohort)s?)\b`.
  A lower bound: abstracts omit most data statements.

## Analyses

Uncertainty: stratified bootstrap over papers (resampling within group and
venue-year), 2000 resamples, seed 20261017, percentile 95% intervals.

* **R-A1 (primary).** Mantel-Haenszel risk difference in O1, medical minus
  other, over the six venue-years; also each venue-year separately. Label
  "medical papers are less often released with code" only if the pooled CI
  upper bound is below 0 AND the difference is negative in at least four of the
  six venue-years. If the label is not met, the abstract does not claim a
  readiness difference across venues.
* **R-A1 secondary and sensitivity (reported, never substituted):** clinical
  only; life science only; O2 instead of O1; topic-only label (candidates by
  topic list, no coding) versus non-candidates; excluding ICML 2026.
* **R-A2 (descriptive).** The same estimate for O3. No label.
* **R-A3 (descriptive, ICML 2026).** Join ICML 2026 papers to the challenge by
  OpenReview id = `orid`. Attempted = at least one judged logbook after
  deduplication, excluding Kate's namespace `kondratevakate`. Decided = at
  least one verified or falsified claim verdict in any logbook. Reported:
  decided share in the four cells medical/other by O1 yes/no; the medical
  minus other difference in decided share, unadjusted and Mantel-Haenszel
  adjusted over O1 strata. The verdicts were seen in protocol 6; O1 has not
  been linked to them before this protocol. Sensitivity: protocol 6 label.
* **R-B1 (descriptive, small n).** Both coders label all 100 RECLAIM papers
  (title and abstract from arXiv), consensus with adjudication. Reported by
  group: number of papers, distribution of the dataset's `tier` field as
  released, and median `h100_hours_adjudicated`. Per-paper agent outcomes are
  reported by group only if the RECLAIM repository releases them; otherwise
  stated as not available.

## Evidence boundary

* A code link is a declared link, not working code; data access is seen only
  in abstracts. Readiness here is a lower bound on what a third party needs.
* Virtual-site metadata are entered by authors and venues and may be
  incomplete or added at different times per venue; the stratified estimate
  compares groups within a venue-year only.
* Labels come from language-model coders, validated against each other, not
  against domain experts.
* Observational and descriptive; no causal effect of being medical is claimed.
* Only aggregates and per-paper labels keyed by public ids are committed; no
  author is named.

## Amendment 1 (2026-10-18, after download, before any labelling or readiness computation)

Mechanics only; definitions, outcomes, analyses and labels are unchanged.

* The 2026 virtual-site files (ICLR 2026, ICML 2026) have no `abstract` field;
  their link fields (`url`, `eventmedia`) are present. The first build of the
  paper table therefore excluded both strata as "no abstract". Abstracts are
  added as follows, and units whose abstract cannot be found are excluded and
  counted:
  - ICML 2026: from the local `papers.json` (key `abstracts`, keyed by
    OpenReview id; SHA-256 recorded in the manifest), joined on the forum id.
  - ICLR 2026: from each paper's virtual-site page
    `https://iclr.cc{virtualsite_url}`, the text of the element whose id starts
    with `abstract`, fetched at most two requests per second.
* Coding in batches: no single subagent can read several thousand abstracts.
  Coder 1 and coder 2 are each a role played by successive Claude subagent
  instances with the identical prompt, each instance coding one batch of at
  most 250 items. Batches follow the shuffled sheet order. Instances of one role
  never see the other role's files.

## Amendment 2 (2026-10-18, before any labelling or readiness computation)

Approved by Kate in chat. Scope extended to computer-vision venues and to
NeurIPS 2026. No paper had been labelled and no readiness share computed.

* Strata (12): ICLR 2025, ICLR 2026, NeurIPS 2024, NeurIPS 2025, NeurIPS 2026,
  ICML 2025, ICML 2026 (machine learning); ICCV 2025, CVPR 2024, CVPR 2025,
  ECCV 2024, ECCV 2026 (computer vision). Sources: the same virtual-site files
  at `iccv.thecvf.com`, `cvpr.thecvf.com`, `eccv.ecva.net` and `neurips.cc`.
  CVPR 2026 is excluded because its virtual-site file lists only 200 of its
  papers (the CVF list has no code-link field). ICCV 2026 does not exist.
* Abstracts for NeurIPS 2026 and ECCV 2026 (files without an `abstract` field)
  come from each paper's virtual-site page, as for ICLR 2026 in Amendment 1.
* Unit: accepted items with `eventtype` Poster, Oral or Spotlight (ECCV 2026
  uses Spotlight). Computer-vision files have no OpenReview link, so the
  deduplication key falls back to `sourceid`, else `id`. They have no
  health or biology topics, so candidates there come from the regex only.
* R-A1 label rule restated for 12 strata: the pooled CI upper bound below 0
  AND a negative difference in at least 8 of the 12 strata (two thirds, as in
  four of six). Added secondary analyses: machine-learning and
  computer-vision venues separately; 2024-2025 strata only (excluding all
  2026 strata, whose code links may still be added before or after the
  meeting).
* All candidates are coded, as frozen (option b).

# Protocol 8: hidden science — what a crowd can check without the data

Frozen 2026-10-19, approved by Kate in chat, before any sample was drawn,
any full text fetched or any contrast computed. Designed after the
results of protocols 3, 4, 6 and 7 were seen. Every analysis below uses data
already looked at in aggregate (ICML 2026 verdicts, medical labels); none of
the contrasts below has been computed. Part of the PSE9 abstract
"Unfalsifiable by default", extended towards hidden science (research whose
evidence rests on data others cannot access) and synthetic or toy releases.

## Questions

H8.1 Within papers, is the medical decidability gap concentrated in empirical
     claims rather than theoretical ones?
H8.2 Does a restricted-data mention go with fewer decided papers, and does it
     account for part of the medical gap?
H8.3 What reasons does the judge give when it does not decide a claim, and
     do they differ for medical papers?
H8.4 Asymmetry at scale: are mechanism and theory claims of medical papers
     decided as often as those of other papers, while headline and secondary
     empirical claims are not?
H8.5 Policy test: were papers with at least one experiment on open data
     decided more often? How common are private-only evaluations ("hidden
     science") and released synthetic, sample or toy data?
H8.6 (optional) Expert check of the medical label.

## Units, groups and outcomes

* Papers: ICML 2026 papers in `readiness/data/papers.json` (stratum
  `icml-2026`). Medical = protocol 7 label `clinical` or `life_science`;
  other = protocol 7 non-candidates; candidates labelled `no` are excluded.
  Secondary: clinical only. Sensitivity: protocol 6 label.
* Logbooks: confirmation snapshot (`data/verdicts_confirmation_2026-10-17/verdicts.json`,
  SHA-256 `652f1c16…`), deduplicated as in protocol 1, Kate's namespace
  `kondratevakate` excluded. Attempted paper = at least one logbook.
* Claim verdict: one claim in one logbook. Decisive = `verified` or
  `falsified`. Paper decided = at least one decisive claim verdict.
* Uncertainty: paper-cluster bootstrap (resampling papers within group),
  2000 resamples, seed 20261017, percentile 95% intervals. Samples drawn with
  seed 20261018.

## H8.1 (primary): theory versus empirical claims within papers

* Claim type by the protocol 4 dictionary (regex in `e1_adjusted/PROTOCOL.md`,
  validated kappa 0.76): `theory` or `empirical`.
* Gap(type) = decisive share, medical minus other, among claim verdicts of
  that type in attempted papers. Estimate: DiD = Gap(empirical) minus
  Gap(theory).
* Label "the medical gap is concentrated in empirical claims" only if the DiD
  CI upper bound is below 0. Reported with both gaps, their CIs and counts.

## H8.2 (descriptive): restricted-data mentions

* Restricted-data mention = protocol 7 outcome O3 (title or abstract regex).
* Decided share among attempted papers in the four cells medical/other by O3;
  medical minus other gap unadjusted and Mantel-Haenszel adjusted over O3.

## H8.3 (descriptive): the judge's stated reasons

* Sample: 150 `toy` or `inconclusive` claim verdicts from attempted medical
  papers and 150 from attempted other papers, at most 3 per paper.
* Two blind coders (Claude subagents) code all 300 from the claim text and the
  judge's evidence text; paper titles are masked and the group is hidden.
  Reason (one, the main one): `data_unavailable` (restricted, private or
  unreleased data), `artifacts_missing` (weights, predictions, outputs or
  derived data not released), `code_missing_or_broken`,
  `reduced_scale_or_compute`, `not_attempted`, `underspecified_paper`,
  `other`. Consensus by agreement, else adjudication by this session with a
  written reason. Cohen's kappa reported.
* Reported: share of each reason by group with Wilson intervals, and the
  medical minus other difference for `data_unavailable` and
  `artifacts_missing` with bootstrap CIs.

## H8.4 (secondary): claim-type map and the asymmetry

* Codebook: protocol 3 claim types (`theory`, `mechanism`, `data_protocol`,
  `headline_performance`, `secondary_empirical`), from claim text only.
* Items: every unique claim text of attempted medical papers, and a random
  sample of the same number of unique claim texts from attempted other papers.
* Coding as protocol 7 option (b): coder 1 codes all, coder 2 a random 25%,
  gate kappa at least 0.80 (else coder 2 codes all and every disagreement is
  adjudicated). Claude subagents; model recorded from transcripts.
* Decisive share by group and type at the claim-verdict level. Contrast:
  Gap({headline_performance, secondary_empirical}) minus
  Gap({theory, mechanism}).
* Label "checkable-without-data claims were decided as often, data-dependent
  claims were not" only if the CI of Gap({theory, mechanism}) includes 0 AND
  the contrast CI upper bound is below 0.

## H8.5 (policy test and prevalence): open data, private data, synthetic releases

* Papers: every attempted medical paper and a random equal-size sample of
  attempted other papers.
* Full text: arXiv HTML at `https://arxiv.org/html/<arxiv id>` (id from
  `papers.json`), else the arXiv PDF parsed with `pypdf`; at most one request
  per three seconds. Papers without full text are excluded and counted.
* Excerpt: paragraphs matching, case-insensitive,
  `dataset|data availab|publicly available|public|private|proprietary|in-house|institutional|IRB|ethic|consent|synthetic|simulated|toy|sample data|demo|download|access|PhysioNet|credential|request|licen|weights|checkpoint|code`,
  in document order, capped at 6,000 characters, with the title.
* Two blind coders (Claude subagents) code every paper (group hidden):
  - `main_data_access`: `open` (downloadable without a data-use agreement),
    `credentialed_or_request` (e.g. PhysioNet credentialing, UK Biobank
    application, on request), `private` (not released), `synthetic_only`,
    `unclear`;
  - `open_data_experiment`: `yes` if at least one main experiment uses open
    data, `no`, `unclear`;
  - `synthetic_or_sample_release`: `yes` if the paper states a released
    synthetic, sample, demo or toy dataset for running the code, `no`,
    `unclear`;
  - `weights_release`: `yes`, `no`, `unclear`.
  Kappa per field; consensus by agreement, else adjudication with a reason.
* Reported: prevalence of each field by group; hidden science = medical
  papers with `main_data_access` = `private` and `open_data_experiment` = `no`.
  Decided share by `open_data_experiment` within medical and within other
  papers; Mantel-Haenszel difference (yes minus no) over the two groups.
* Label "papers with an open-data experiment were decided more often" only if
  the pooled CI lower bound is above 0. `unclear` is excluded and counted.

## H8.6 (optional): expert check

If Kate provides it: 60 ICML 2026 papers drawn at random (30 protocol 7
medical, 30 non-candidates), coded blind by Kate with the E0 codebook.
Agreement with the protocol 7 labels is reported; labels are not changed.

## Evidence boundary

* All verdicts come from one LLM judge. Decisive is the challenge's label.
* Observational: teams chose papers and claims; the within-paper contrasts
  (H8.1, H8.4) remove paper choice but not claim difficulty.
* Excerpts can miss data statements; `unclear` is reported, not imputed.
* Coders are language models; agreement is between model coders.
* The policy test describes existing variation; it does not show what a
  mandate would do.
* A synthetic or transfer reproduction pilot (re-running private-data papers
  on open data or schema-faithful synthetic data) is out of scope here and
  would be a separately frozen protocol.
* Aggregates and per-item codes keyed by public ids only; no author named.
