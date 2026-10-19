# Protocol 6: how decidable were medical papers under crowd reproduction?

**Question.** How often did the challenge reach any decisive verdict (verified or falsified) on medical and
life-science papers, compared with other papers?

**Protocol.** [`PROTOCOL.md`](PROTOCOL.md), frozen in commit `15c5441` before the candidate sheet was drawn
or coded. Descriptive: no causal claim and no prespecified label.

**Inputs.** The verdict snapshot `../../data/verdicts_confirmation_2026-10-17/verdicts.json` (SHA-256
`652f1c16…`); `papers.json` (6,341 ICML 2026 papers) and the 255 candidates of `medical_neuro_papers.csv`,
both from the challenge's dataset at the repository root (not committed); the blind candidate sheet
[`sheets/candidates.csv`](sheets/); codes from two Claude subagent coders (`codes_A.csv`, `codes_B.csv`) and
[`adjudication.csv`](adjudication.csv).

**Run.**

```bash
cd verdict_audit/falsifiability
python analyze6.py agreement   # kappa and the disagreement list (adjudicated into adjudication.csv)
python analyze6.py results     # D1-D5 with bootstrap intervals -> results6.json
```

**Outputs.** [`results6.json`](results6.json) (attempt rate, fully verified share, undecided share, verdict
mix, teams needed; with and without the author's logbooks); run log [`results_run.log`](results_run.log).

**Used in.** Abstract 2: attempt rate 33% versus 33%; fully verified 1.5% versus 20.4%; no decisive verdict
61.8% versus 35.1% (`without_kate`).

**Limits.** One LLM judge; teams chose which papers to attempt; labels from language-model coders (kappa 0.97).
