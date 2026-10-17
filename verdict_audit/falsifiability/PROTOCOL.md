# Protocol 6: how decidable were medical ICML 2026 papers under crowd reproduction?

Frozen 2026-10-18 before the candidate sheet was drawn or coded. Git commit is the
provenance. Descriptive study for the PSE9 abstract "Unfalsifiable by default".

## Status of the evidence

The quantities below were looked at in an exploratory pass with an unvalidated
label (2% of 90 attempted medical papers fully verified versus 20% of 2,086
others; 9% versus 31% with three or more teams). This protocol re-estimates them on
a validated label with intervals. It is descriptive: no causal claim, and the
adjusted verdict gap of protocol 4 (E1, CI including zero) stands.

## Data

`data/verdicts_confirmation_2026-10-17/verdicts.json` (SHA-256
`652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526`).
Deduplication as in protocol 1 (one logbook per reproducer and paper). Papers:
`papers.json` (6,341). Team = Space namespace.

## Groups

* Candidates: the 255 papers of `medical_neuro_papers.csv`.
* Coding: two blind Claude subagent coders label each candidate from title and
  abstract with the E0 codebook (`clinical`, `life_science`, `no`); this session
  adjudicates disagreements with a written reason. Agreement is reported.
* **Medical** = candidates with consensus `clinical` or `life_science`.
  Secondary: `clinical` only.
* **Other** = papers in neither `medical_neuro_papers.csv` nor
  `audit/public_repro_audit.csv` (the broad list, whose NPV was 0.99 in E0).
* Papers in the broad list but not in the candidates, and candidates coded `no`,
  are excluded from both groups (ambiguous), and their number is reported.

## Quantities (per group)

* D1 attempt rate: share of papers with at least one judged logbook.
* D2 depth, among attempted papers: mean teams; share with at least 2 and at least
  3 teams.
* D3 decidability, among attempted papers: share where at least one logbook
  verified every claim; share with at least one falsified claim in any logbook;
  share where no logbook gave any claim a verified or falsified verdict
  (undecided).
* D4 claim-level verdict mix (verified, falsified, toy, inconclusive).
* D5 expected reliability of a paper's crowd verdict, by Spearman-Brown with
  each paper's observed number of teams and a single-team reliability of 0.23
  (protocol 1 discovery kappa) and 0.15 (confirmation kappa): mean over attempted
  papers. Also the number of teams needed for 0.7 and 0.8.

Intervals: paper-level bootstrap, 2,000 resamples, seed 20261017, for D1 to D4
and for medical minus other differences. Every quantity is reported with and
without Kate's own logbooks (namespace `kondratevakate`), because she reproduced
many medical papers; the version without her is primary.

## Evidence boundary

* Verdicts come from one LLM judge; "verified" is the challenge's label, not a
  confirmed truth.
* Spearman-Brown treats kappa as single-team reliability and teams as parallel;
  copying between teams (studied separately) would make the true reliability
  lower.
* Reproducers chose papers; differences describe the challenge, not a causal
  effect of being medical.
