# Unfalsifiable by default: medical papers in the ICML 2026 agent reproduction challenge

DRAFT abstract, 2026-10-19, extended with protocol 8 (hidden science and
synthetic releases). Target: PSE9 (Copenhagen 2027), second abstract next to
the voters study. The submission format is not yet known. Every number is
traced to a file in Appendix A.

## Abstract

Agentic crowd reproduction promises to check machine-learning claims at
conference scale. In the ICML 2026 challenge, medical and life-science papers
were attempted as often as other papers (33% versus 33%) but were rarely
decided. On a label validated by two blind coders (207 papers), 1.5% of
attempted medical papers had a logbook verifying every claim, against 20.4% of
other papers, and 61.8% received no decisive verdict at all, against 35.1%. The
judge was not harsher on them: within the same team and claim type, clinical
claims were verified 3.3 percentage points less often (95% CI −9.9 to +3.2).

Missing code does not explain the gap. Across 50,185 papers at six venues
(2024–2026), medical papers declared a code link more often (24.5% versus
16.9%; within-venue difference +7.4 points, CI +5.8 to +9.0), and adjusting for
it left the ICML 2026 gap unchanged (−18.6 versus −19.1 points). Hidden
science, research resting on data that no one else can access, does not
appear to explain it either, although our data cannot rule it out. Such papers
were not avoided: medical papers whose abstract mentioned restricted data were
attempted more often than other medical papers (20 of 25 versus 30%), although
one team took 11 of the 20. In the full texts of 176 attempted papers, only 1
of 79 medical papers was clearly evaluated on private data alone (27 were
unclear); 18% used credentialed or on-request data, against 1% of other papers.
We found no detectable difference in how often medical papers on restricted
and on open data were decided (47% versus 50%), but the comparison is small
(15 versus 36 papers; CI of the difference −33 to +27 points). When the judge left a medical claim undecided, it gave unavailable data
as the reason in 12% of cases (other papers 1.3%), and an experiment that was
not run or was run at reduced scale in 66% (other papers 70%). The gap also
reached claims that need no private data: claims that a method works as
described were decided 22 points less often than in other papers (CI −38 to
−1).

Synthetic, sample or toy data for running the code were stated in only 2 of
79 medical papers. Requiring such releases would start almost from zero, and
might not by itself make these papers decidable: the judge already has a
"toy" verdict, which is neither verified nor falsified. Venues that ask for
synthetic or toy data should state which claims such data can settle (a
method's mechanics, not its headline numbers). Reproduction efforts should
report the share of papers they could not decide, not only the share they
verified.

(about 435 words)

## Notes for the authors (not part of the abstract)

* "Unfalsifiable" rests on decidability in the challenge (protocol 6), not on
  failed reproductions. Two mechanisms were tested and neither explains the
  gap: code links (protocol 7; the prespecified label is not met and the
  difference runs the other way) and data access (protocol 8).
* No prespecified label of protocol 8 is met. H8.1: the gap is not
  concentrated in empirical claims (difference in differences +0.8 points,
  CI −14.0 to +19.9). H8.4: checkable claims were not decided as often as in
  other papers (theory and mechanism together −24.1 points, CI −40.0 to −6.8).
  H8.5: papers with an open-data experiment were not decided more often; the
  pooled difference was −24.3 points (CI −39.1 to −9.3), the opposite
  direction.
* The H8.5 reversal is composition, shown post hoc (Note 8-2, PH8-a): in other
  papers the no-open-data cell has 22 papers, of which 13 are synthetic-only
  and 10 have no empirical claim (the two overlap). Among empirical,
  non-synthetic papers the difference is −14.9 points (CI −40.2 to +9.5). The
  restricted versus open comparison (PH8-b) and the attempt rates (PH8-d,
  Note 8-3) in the abstract are also post hoc and have no label.
* The two claim classifiers disagree on theory claims: the protocol 4
  dictionary gives a theory gap of −18.5 points (CI −39.2 to −1.6), the H8.4
  coders −2.2 points (CI −28.5 to +16.5). The abstract uses neither. Headline
  performance claims: −14.5 points (CI −25.0 to −2.9).
* Hidden science is counted among attempted papers only. PH8-d argues against
  avoidance: medical papers whose abstract mentions restricted data were
  attempted in 20 of 25 cases against 90 of 303 (+50 points, CI +33 to +66),
  but one of 16 teams took 11 of the 20, and the abstract regex (O3) misses
  papers that do not name their data. 43 of 220 sampled papers had no arXiv
  id and 1 no usable excerpt.
* "1 of 79" is a lower bound. Excerpts were capped at 6,000 characters, and
  adjudication used the excerpt only, so a named institutional dataset without
  stated access terms (e.g. a hospital dataset) was coded `unclear`, not
  `private`; 27 of 79 medical papers have `unclear` data access.
* Bootstrap intervals for H8.3, H8.5 and PH8 were recomputed after a
  hash-order bug (Note 8-4); the values first committed in 0e5ccb7 are
  superseded. Point estimates did not change.
* The release fields (synthetic or sample data, weights) had low agreement
  (kappa 0.12 and 0.40) and most excerpts reached the cap: "2 of 79" (and 4
  of 79 for weights) are lower bounds. A code release that runs synthetic
  experiments was not counted as a synthetic-data release (Note 8-1).
* Medical papers on open data were decided in 50% of cases (95% Wilson
  interval 34–66%), other papers on open data in 63% (50–74%); post hoc,
  without a test.
* Restricted-data mentions in abstracts are seven times as common in medical
  papers (7.6% versus 1.1%); adjusting the ICML 2026 gap for them moves it
  from −19.1 to −15.3 points (CI −25.8 to −4.9).
* In the author's release audits of 14 medical papers with released code, 62%
  of the claims that could not be fully checked lacked intermediate artifacts
  or weights. These papers were chosen by interest and feasibility.
* The code-link field is author-declared metadata, not working code. NeurIPS
  2026 and ECCV 2026 have few links filled in yet (3.5% and 4.7% of medical
  papers); the 2024–2025 strata alone give +9.1 points (CI +6.7 to +11.6).
* RECLAIM (100 NeurIPS 2025 papers) contains only 6 medical papers: 2 of 6 were
  reproduced clean by at least one agent, against 33 of 94 others. Context
  only.
* Two medical labels are used. The 207-paper label (protocol 6) gives the
  decidability figures; the protocol 7 label (110 attempted medical papers in
  ICML 2026) gives the adjusted gaps and all protocol 8 figures.
* All verdicts come from one LLM judge (`zai-org/GLM-5.2`). Labels and codes
  come from language-model coders: Claude subagents for protocol 6 (kappa
  0.97), `claude-sonnet-5-5` for protocol 7 (kappa 0.96, gate 0.80) and
  protocol 8 (judge reasons kappa 0.89; claim types 0.90, gate 0.80; data
  access 0.94; open-data experiment 0.92). Analyses are observational.
* Disclosure: the author took part in the challenge and wrote the release
  audits. Challenge analyses exclude the author's logbooks.

## Appendix A: provenance of numbers

| Number | File (in `verdict_audit/`) | Key |
|---|---|---|
| Attempt rate 33% vs 33% (0.3285, 0.3319) | `falsifiability/results6.json` | `without_kate.{medical,other}.D1_attempt_rate` |
| 207 medical papers; kappa 0.97 | `falsifiability/results6.json` | `medical_n`, `agreement.kappa` |
| Fully verified 1.5% vs 20.4% | `falsifiability/results6.json` | `without_kate.*.D3_fully_verified` |
| No decisive verdict 61.8% vs 35.1% | `falsifiability/results6.json` | `without_kate.*.D3_undecided` |
| Clinical −3.3 pp (−9.9 to +3.2) | `e1_adjusted/results_e1_validated.json` | `clinical.without_kate` |
| 50,185 papers; 2,606 medical vs 43,653 other | `readiness/results/readiness.json` | `units`, `groups` |
| Code link 24.5% vs 16.9%; +7.4 pp (5.8–9.0); 11 of 12 | `readiness/results/readiness.json` | `R_A1` (`negative_strata` = 1) |
| 2024–2025 only +9.1 pp (6.7–11.6) | `readiness/results/readiness.json` | `R_A1_secondary.strata_2024_2025` |
| NeurIPS 2026 3.5%, ECCV 2026 4.7% (medical) | `readiness/results/readiness.json` | `R_A1.per_stratum` |
| Decidability gap −19.1 unadjusted, −18.6 adjusted | `readiness/results/readiness.json` | `R_A3.protocol7_label` |
| Restricted-data mentions 7.6% vs 1.1% | `readiness/results/readiness.json` | `R_A2` |
| Gap adjusted for restricted-data mention −15.3 (−25.8 to −4.9) | `readiness/results/p8.json` | `H8_2.adjusted`, `adjusted_ci` |
| 110 attempted medical ICML 2026 papers | `readiness/results/p8.json` | `attempted.medical` |
| 176 full texts; 79 medical, 97 other; 43 no arXiv id, 1 no excerpt | `readiness/results/p8.json` | `H8_5.prevalence.*.n`, `H8_5.fulltext_status` |
| Private alone 1 of 79 | `readiness/results/p8.json` | `H8_5.hidden_science.n` |
| Credentialed or on request 18% (14/79) vs 1% (1/97); unclear 27/79 | `readiness/results/p8.json` | `H8_5.prevalence.*.main_data_access` |
| Restricted 47% (7/15) vs open 50% (18/36); −3 pp (−33 to +27) | `readiness/results/p8_posthoc.json` | `PH8_b` |
| Open data: medical 50% (34–66), other 63% (50–74) | `readiness/results/p8_posthoc.json` | `PH8_b.cells` |
| Attempted 20 of 25 (O3) vs 90 of 303 (30%); +50 (+33 to +66); 16 teams, one with 11 | `readiness/results/p8_posthoc.json` | `PH8_d` |
| Data unavailable 12% vs 1.3%; not run or reduced scale 66% vs 70% | `readiness/results/p8.json` | `H8_3.shares` (`not_attempted` + `reduced_scale_or_compute`) |
| Method works as described −22 pp (−38 to −1); headline −14.5 (−25.0 to −2.9) | `readiness/results/p8.json` | `H8_4.by_type.{mechanism,headline_performance}` |
| Theory gap −18.5 (dictionary) vs −2.2 (coders) | `readiness/results/p8.json` | `H8_1.gap_theory`, `H8_4.by_type.theory` |
| Synthetic or sample release 2 of 79; weights 4 of 79 | `readiness/results/p8.json` | `H8_5.prevalence.medical.{synthetic_or_sample_release,weights_release}` |
| H8.1 DiD +0.8 (−14.0 to +19.9) | `readiness/results/p8.json` | `H8_1.did`, `did_ci` |
| H8.4 checkable −24.1 (−40.0 to −6.8) | `readiness/results/p8.json` | `H8_4.gap_checkable`, `gap_checkable_ci` |
| H8.5 policy −24.3 (−39.1 to −9.3) | `readiness/results/p8.json` | `H8_5.policy_mh_diff`, `policy_mh_ci` |
| PH8-a 22 papers; 13 synthetic-only, 10 without empirical claim | `readiness/results/p8_posthoc.json` | `PH8_a.other_open_0` |
| PH8-c −14.9 (−40.2 to +9.5) | `readiness/results/p8_posthoc.json` | `PH8_c.mh_diff`, `mh_ci` |
| Protocol 8 kappas 0.89, 0.90, 0.94, 0.92, 0.12, 0.40 | `readiness/results/p8_coding.json` | `h83`, `h84`, `h85.agreement` |
| "toy" verdict label | `../data/verdicts_confirmation_2026-10-17/verdicts.json` (repository root) | verdict categories: verified, falsified, toy, inconclusive |
| 62% artifact-blocked; 14 papers | `healthcare/results.json` | `resolved.Q1` |
| Label kappa 0.96 (protocol 7) | `readiness/results/labelling.json` | `gate.kappa` |
| RECLAIM 6 medical; 2 of 6 vs 33 of 94 | `readiness/results/readiness.json` | `R_B1` |
