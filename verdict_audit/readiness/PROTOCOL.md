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
