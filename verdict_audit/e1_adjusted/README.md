# Protocol 4: medical label check (E0) and the adjusted medical verdict gap (E1)

**Question.** Are medical claims in the ICML 2026 challenge judged more severely, once the reproducing team
and the claim type are held fixed?

**Protocol.** [`PROTOCOL.md`](PROTOCOL.md), frozen in commit `26e69c6` before any sample was drawn, coded or
modelled. Amendment 1 (`cf58958`): a fixed-effects logit replaces the conditional logit. Amendment 2
(`9ec6486`): H1 re-estimated on the protocol 6 validated labels.

**Inputs.** The verdict snapshot `../../data/verdicts_confirmation_2026-10-17/verdicts.json` (SHA-256
`652f1c16…`, see [`../README.md`](../README.md)); coding samples drawn by `sample.py` into
[`sheets/`](sheets/) with answer keys in [`keys/`](keys/); blind codes from two Claude subagent coders
(`codes_A_*.csv`, `codes_B_*.csv`) and adjudications (`adjudication_*.csv`).

**Run.**

```bash
cd verdict_audit/e1_adjusted
python sample.py                 # once; draws the E0 and E1 samples (already drawn and committed)
python analyze_e1.py validate    # E0 label validity and dictionary gates -> validation_e1.json
python analyze_e1.py models      # H1 model and sensitivities -> results_e1.json
python e1_validated.py           # Amendment 2 -> results_e1_validated.json
python -m pytest test_e1.py
```

**Outputs.** [`validation_e1.json`](validation_e1.json), [`results_e1.json`](results_e1.json),
[`results_e1_validated.json`](results_e1_validated.json); model log [`models_run.log`](models_run.log).

**Prespecified decision.** "Clinical claims are not judged more severely" holds: within team and claim type,
clinical claims were verified 3.3 percentage points less often (95% CI −9.9 to +3.2), excluding the author's
logbooks (`results_e1_validated.json`, `clinical.without_kate`).

**Limits.** Observational; one LLM judge; labels from language-model coders.
