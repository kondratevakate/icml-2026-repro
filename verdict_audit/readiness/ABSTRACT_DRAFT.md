# Unfalsifiable by default: medical papers in the ICML 2026 agent reproduction challenge

DRAFT abstract, 2026-10-19. Target: PSE9 (Copenhagen 2027), second abstract
next to the voters study. The submission format is not yet known. Every number
is traced to a file in Appendix A.

## Abstract

Agentic crowd reproduction promises to check machine-learning claims at
conference scale. In the ICML 2026 challenge, medical and life-science papers
were attempted as often as other papers (33% versus 33%) but were rarely
decided. On a label validated by two blind coders (207 papers), 1.5% of
attempted medical papers had a logbook verifying every claim, against 20.4% of
other papers, and 61.8% received no decisive verdict at all, against 35.1%. The
judge was not harsher on them: within the same team and claim type, clinical
claims were verified 3.3 percentage points less often (95% CI −9.9 to +3.2).
Missing code does not explain the gap. Across 50,185 papers at ICLR, NeurIPS,
ICML, CVPR, ICCV and ECCV (2024–2026), medical papers declared a code link more
often than other papers (24.5% versus 16.9%; within-venue difference +7.4
percentage points, CI +5.8 to +9.0, positive in 11 of 12 venue-years). In ICML
2026 the decidability gap stayed the same after adjusting for declared code
(−18.6 versus −19.1 points). What differs is access. Medical abstracts
mentioned restricted or private data seven times as often (7.6% versus 1.1%).
In release audits of 14 medical papers that had released code, 62% of the
claims that could not be fully checked lacked intermediate artifacts or
weights. Releasing code is therefore not enough to make medical ML
falsifiable: without data access and released artifacts, a crowd cannot decide
these papers. Reproduction efforts should report the share of papers they
could not decide, not only the share they verified. Venues should ask for
checkable artifacts, not only code links.

(about 290 words)

## Notes for the authors (not part of the abstract)

* "Unfalsifiable" rests on decidability in the challenge (protocol 6), not on
  failed reproductions. Code links were tested as a mechanism and rejected:
  the prespecified label "medical papers are less often released with code"
  (protocol 7) is not met, and the difference runs the other way.
* "Access" is supported by two indirect measures, not proven as a cause.
  Restricted-data mentions in abstracts are a lower bound. The release audits
  are the author's own, on papers chosen by interest and feasibility.
* The code-link field is author-declared metadata, not working code. NeurIPS
  2026 and ECCV 2026 have few links filled in yet (3.5% and 4.7% of medical
  papers); the 2024–2025 strata alone give +9.1 points (CI +6.7 to +11.6).
* RECLAIM (100 NeurIPS 2025 papers) contains only 6 medical papers: 2 of 6 were
  reproduced clean by at least one agent, against 33 of 94 others. The sample
  is too small for the abstract and is reported as context only.
* All verdicts come from one LLM judge (`zai-org/GLM-5.2`). Medical labels come
  from language-model coders: Claude subagents for protocol 6 (kappa 0.97)
  and `claude-sonnet-5-5` for protocol 7 (kappa 0.96, gate 0.80). Analyses are
  observational.
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
| 62% artifact-blocked; 14 papers | `healthcare/results.json` | `resolved.Q1` |
| Label kappa 0.96 (protocol 7) | `readiness/results/labelling.json` | `gate.kappa` |
| RECLAIM 6 medical; 2 of 6 vs 33 of 94 | `readiness/results/readiness.json` | `R_B1` |
