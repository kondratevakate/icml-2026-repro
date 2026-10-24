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

## Amendment 1 (2026-10-24, before any candidate is coded): validation by two blind coders

Kate asked for a validated list. The keyword screen above is kept as a screen; a wider candidate pool is coded.

* Candidates: the union of the 98 screen matches and every paper whose title or abstract matches both an agent/LLM
  term (`agent`, `agentic`, `LLM`, `large language model`) and a research-task term (scientific discovery or
  research, research ideas/papers/pipelines/workflows, hypothesis generation, automated science/research/discovery/
  experiments/review, scientist, paper reproduction or replication, self-driving or autonomous labs, peer review,
  paper or literature-review writing, novelty, ML engineering, Kaggle, the named benchmarks). Pattern in `code.py`
  (`WIDE_A`, `WIDE_B`). About 350 papers; duplicate titles within a venue-year are coded once.
* Two blind coders (claude-sonnet-5-5 subagents, roles as1 and as2), title and abstract only, venue hidden, batches
  of 20, several batches per instance. Prompt in `CODER_PROMPT.md`.
* Labels: `include` (the paper's main subject is a system that carries out steps of scientific research itself —
  ideation, hypothesis generation, experiment design or execution, analysis, paper writing, peer review, or
  reproducing papers — or a benchmark or evaluation of such systems), `exclude` (anything else, including web-search
  "deep research" question answering, general tool-use or coding agents, and scientific question answering without
  research steps), `unclear`. If included, a category: `ai_scientist` (end-to-end research systems),
  `research_step` (one step: ideation, experiments, writing, review), `reproduction` (agents reproducing papers),
  `benchmark` (benchmarks or evaluations of research agents), `lab` (self-driving or autonomous laboratories).
* Cohen's kappa on include/exclude. Every disagreement (label or category) is adjudicated by a claude-opus-5-5
  subagent from the same text, labels in random order, with a reason.
* Outputs replace nothing above: `results/validated.csv`, `results/validated_summary.json` (counts by venue-year and
  category; ICML 2026 attempted and decided; precision of the original keyword screen against the validated label).
