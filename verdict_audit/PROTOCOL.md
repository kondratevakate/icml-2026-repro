# Protocol: reproducibility verdicts in the ICML 2026 reproduction challenge

Frozen 2026-10-17 before running `verdict_audit/analyze.py` on the inputs below.
Git commit and file hashes provide provenance, not external preregistration.
Target: abstract for PSE9 (Perspectives on Scientific Error, Copenhagen, 2027).

## Question

How reliable are judged reproduction verdicts for ML papers, and are medical
and life-science papers reproduced less decisively than other papers? The unit of
evidence is a judged claim in a public reproduction logbook.

## Inputs (discovery snapshot)

| File | SHA-256 | Role |
|---|---|---|
| `verdicts_latest_tmp.json` | `b4bae4eb8a333dc98c3edd6865452ff96009b5b1a6dd1685d96a61bdaf31c239` | Primary verdicts, judged 2026-07-08 to 2026-07-28 |
| `verdicts_live.json` | `76eeb935cc0972ec5b0ef9464a43d0920e00af1328db5460a71b61971351d544` | Earlier snapshot, used only for judge test-retest |
| `claims_anchored.json` | `eb3f2d878646ca5c40da121741a613c1f9e1e10c14845f373db107e74a4ef439` | Anchored claim texts |
| `claims.json` | `af5ab2d62f786ae36861957cbd08b4188f6d4c86e67152becc661a9c5bbb9d57` | Loose claim texts |
| `audit/public_repro_audit.csv` | `459ae5d58aaeae2c777870e664e49920eb2d9aa75052201a9d9ea85e51addcd8` | Medical/life-science label (703 papers) |
| `medical_neuro_papers.csv` | `92dc7e4cb51bef127428975c7372fd05f288df0566eff6454275887e68eeabe9` | Narrower medical label, sensitivity only |

The script refuses to run if any hash differs.

## Definitions

* Logbook: one record keyed by `space_id`. Reproducer: the namespace before `/`.
  Paper: `orid`.
* Deduplication: one logbook per (reproducer, paper); keep the latest `judged_at`.
  Ties break by `space_id` order. The number dropped is reported.
* Verdict categories: verified, falsified, toy, inconclusive. Any other value is
  `other` and counted. Decisive means verified or falsified.
* Text normalization: Unicode NFKC, casefold, whitespace collapsed, stripped.
  Claim identity: (paper, normalized anchored text). Primary analysis uses
  normalized matching; exact matching is a sensitivity analysis.
* Medical label: paper `orid` in `audit/public_repro_audit.csv`. This label comes
  from area tags plus a manual shortlist and is not manually validated per paper.
* Agreement: Fleiss-type kappa with pooled marginals. For claim i with n_i>=2
  ratings, P_i = sum_k n_ik(n_ik-1) / (n_i(n_i-1)); Po = mean of P_i, each claim
  weighted equally; Pe = sum_k p_k^2, where p_k pools all ratings on included
  claims; kappa = (Po-Pe)/(1-Pe).
* Uncertainty: cluster bootstrap over papers (C2, C3) or over logbooks (C4), 2000
  resamples, percentile 95% intervals. Permutation tests use 2000 permutations.
  Seed 20261017 for all resampling.

## Claims under test

**C1 Anchoring (descriptive).** Share of judged claims in deduplicated logbooks
whose text matches an anchored claim of the same paper; share matching only the
loose `claims.json` text; share matching neither. No direction prespecified.

**C2 Inter-reproducer agreement (primary).** Among anchored claims judged by at
least two distinct reproducers: Po, Pe, kappa with paper-cluster CI; the share of
claims with at least one verified-versus-falsified conflict; kappa on the binary
decisive-versus-not outcome. Negative control: shuffle verdict labels across the
included ratings while keeping each claim's rating count; Po under the null must
centre on Pe. Report the permutation p-value for observed Po.

**C3 Medical gap.** Difference, medical minus other, in the share of claims judged
verified, inconclusive, toy and falsified, over all judged claims in deduplicated
logbooks. Paper-cluster bootstrap CI. Mutation test: permute the medical label
across papers (number of medical papers fixed); report two-sided permutation
p-value for the verified-share difference. Secondary: C2 kappa within medical and
within other papers.

**C4 Judge test-retest (control for C2).** Logbooks present in both snapshots
with identical `sha` and different `judged_at` were judged twice on the same
content. Pair claims by position when the normalized texts agree. Report Po,
kappa with logbook-cluster CI, and the verified-versus-falsified flip rate.
C2 is interpreted relative to C4: inter-reproducer disagreement that does not
exceed judge test-retest disagreement cannot be attributed to the reproductions.

## Prespecified interpretation

* C2 is called "substantial disagreement between reproducers" only if the upper
  95% bound of kappa is below 0.60 AND the C2 kappa CI lies entirely below the
  C4 kappa CI. Otherwise report the numbers without that label.
* C3 is called a "medical gap in verified share" only if the bootstrap CI excludes
  zero and the permutation p-value is below 0.05. One primary test (verified
  share); the other three shares are descriptive.
* The 0.60 threshold is this protocol's own convention, fixed now.

## Sensitivity analyses (reported, never substituted for primary)

1. Exclude the organiser namespace `ICML-2026-agent-repro`.
2. Exact instead of normalized claim-text matching.
3. Medical label restricted to `medical_neuro_papers.csv`.
4. No deduplication (all logbooks).

## Confirmation

A later verdict snapshot, restricted to logbooks first judged after 2026-07-28,
is a disjoint confirmation set. Downloading it requires a separate decision by
Kate. Code and this protocol stay unchanged; any change is documented and the
discovery result is not relabelled as confirmed.

## Evidence boundary

* All verdicts come from one LLM judge (`zai-org/GLM-5.2`), not from human
  experts. Verdict reliability here means reliability of judged logbooks.
* Many logbooks were produced by AI agents. Results describe this challenge's
  agentic reproduction ecosystem, not reproducibility of ML papers in general.
* Reproducers choose which papers to attempt. Medical-versus-other contrasts are
  observational and confounded by paper difficulty, compute and data access.
* Only aggregate results are reported. No individual participant is named or
  ranked in outputs, apart from the author's own logbooks.
* Repository availability numbers from `audit/` are not used: only 50 of 129
  GitHub repositories were checked live because of rate limits.

## Amendment 1: confirmation run (frozen 2026-10-17, after discovery, before confirmation)

Kate approved downloading the current snapshot. Discovery results were committed
before this amendment and are not relabelled.

* Input: `data/verdicts_confirmation_2026-10-17/verdicts.json`, SHA-256
  `652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526`, downloaded
  2026-10-17 from `ICML-2026-agent-repro/verdicts` (6,885 logbooks, judged up to
  2026-08-03, one judge model). Not versioned; reproducible from the dataset
  history.
* Confirmation set: logbooks whose `space_id` appears in neither discovery
  snapshot AND whose `judged_at` is later than 2026-07-28T12:51:33+00:00
  (1,910 logbooks, 997 papers). The protocol says "first judged after"; the
  snapshot keeps only the latest judgement per Space, so absence from both
  earlier snapshots is the operational definition.
* Deviation: `analyze.py` hard-codes the discovery inputs and refuses to overwrite
  `results/`. `confirm.py` imports its analysis functions unchanged and applies
  them to the confirmation set; deduplication, anchored texts (the frozen local
  `claims_anchored.json`), medical labels, seeds, resamples and thresholds are
  identical. Agreement (C2) uses only ratings from confirmation logbooks.
* C4 replication: Spaces present in the 2026-07-28 snapshot and the new snapshot
  with identical `sha` and a later `judged_at` (different judgement events from
  discovery C4; the same logbook content may recur).
* Confirmation succeeds for a label only if the same prespecified rule holds on
  the confirmation set. A failed confirmation is reported as such.
