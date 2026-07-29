# PEQ-Net Publication Preflight

Date: 2026-07-29.

## Target

- Space:
  `kondratevakate/repro-smooth-multi-policy-causal-effect-estimation-in-longitudinal-settings`
- Paper tag: `paper-bIcz7bIZSo`
- arXiv metadata: `2605.14284`
- Prepared score: `4/12`
- Conservative expected score: `4/12`

## Frozen verdicts

| Claim | Verdict | Independent evidence |
| --- | --- | --- |
| C1 | `INCONCLUSIVE` | No code or processed MIMIC-III; printed DGP singular at `i=1`. |
| C2 | `INCONCLUSIVE` | Same blockers; expanded DGP delegates to singular equations. |
| C3 | `INCONCLUSIVE` | Dynamic policies are specified but model and DGP are not executable. |
| C4 | `VERIFIED WITH QUALIFICATION` | Independent MMD plus metric-MDS pipeline; ordering correlation `1.0`. |
| C5 | `FALSIFIED AS A LIPSCHITZ GUARANTEE` | Pair-specific lemma constant, MDS mismatch, targeting counterexample. |
| C6 | `INCONCLUSIVE` | Exact restricted-data cohort path is unavailable. |

## Checks

- [x] arXiv source and PDF hashes frozen.
- [x] No patient-level data included.
- [x] No official paper source or figures included in the publishable bundle.
- [x] Independent audit emits all six verdicts.
- [x] Four unit tests pass.
- [x] Trackio logbook contains executive summary, six claim pages, and conclusion.
- [x] Executive summary and poster are pinned.
- [x] Every claim page states the paper claim and verdict.
- [x] Local `validate_icml_logbook.py` passes for the exact target Space.
- [x] Reproducibility bundle hashes are frozen.
- [x] No external reproduction attempts or participant verdicts were used.

## Known limitations

- C4 verifies that the representation pipeline is operational on a controlled
  policy family, not that the downstream neural model reproduces Table 1 or 2.
- The C5 counterexample directly falsifies a proof implication used to obtain
  the theorem. A repaired theorem could hold under additional uniform
  embedding and targeting-map assumptions.
- C6 must not be upgraded from `INCONCLUSIVE` based on a visually similar
  cohort or the paper figure alone.

## Publication state

- Hub Space:
  `https://huggingface.co/spaces/kondratevakate/repro-smooth-multi-policy-causal-effect-estimation-in-longitudinal-settings`
- Canonical static host:
  `https://kondratevakate-repro-smooth-multi-policy-causal-02b1dfc.static.hf.space`
- Remote commit:
  `aff8cc40ec261e509ea66be93d2c019449c28ed0`
- Runtime: `RUNNING`, static SDK.
- Remote tags: `paper-bIcz7bIZSo`, `arxiv:2605.14284`.
- Remote index, executive summary, C4, C5, and conclusion hashes match local
  files exactly.
