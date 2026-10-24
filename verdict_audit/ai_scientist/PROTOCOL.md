# AI-scientist papers: screening rule (frozen 2026-10-24, before any paper was screened)

Kate asked to find, among all paper sets of this repository, the papers about AI scientists and autonomous research
agents, on a branch of their own (`ai-scientist-papers`), separate from the PSE9 / Protocol 9 work.

## Units

All 50,185 papers in `verdict_audit/readiness/data/papers.json` (ICLR 2025-2026, ICML 2025-2026, NeurIPS 2024-2026,
CVPR 2024-2025, ICCV 2025, ECCV 2024 and 2026; title and abstract). The P9 frame is a subset of these, so it is
covered. Full texts are not screened.

## Inclusion rule

A paper is included if its title or abstract matches, case-insensitively, at least one of these patterns
(`screen.py`, `PATTERNS`), which name a system that carries out research itself or a benchmark of such systems:

* AI scientist (`ai[- ]scientist`, `ai co-?scientist`, `automated scientist`, `autonomous scientist`, `scientist agent`)
* research agent (`research agents?`, `(ml|machine learning|scientific|science) agents?` followed within a few words
  by research, experiment or discovery is not used; only the fixed phrases listed in the script)
* automated or autonomous research (`automated research`, `autonomous research`, `auto-?research`,
  `research automation`, `automat(e|ing|ion of) (scientific )?research`, `agentic research`, `end-to-end research`)
* automated discovery (`automated scientific discovery`, `autonomous scientific discovery`, `ai-driven scientific discovery`)
* automated experimentation (`self-driving lab`, `autonomous lab`, `autonomous experimentation`)
* agents that reproduce or replicate papers (`reproduc(e|ing|tion) (of )?(ml |research |scientific )?papers`,
  `paper reproduction`, `replicat(e|ing|ion) (of )?(ml |research )?papers`)
* named benchmarks of research agents: PaperBench, MLE-bench, RE-Bench, ScienceAgentBench, CORE-Bench, MLAgentBench,
  ResearchBench, DiscoveryBench

The rule is a screen, not a validated label: its precision and recall are not measured here.

## Outputs

* `results/papers.csv`: venue-year, paper id, title, matched patterns; for ICML 2026, attempted (at least one
  judged logbook in the 2026-10-17 verdict snapshot, author's namespace excluded) and decided (at least one verified
  or falsified claim).
* `results/summary.json`: counts by venue-year; ICML 2026 attempted and decided counts against all ICML 2026 papers.
