# Diabetes safety-generalization publication preflight

Date: 2026-07-29.

## Decision

`PUBLISHED` as a scoped CPU/source reproduction at:

`https://huggingface.co/spaces/kondratevakate/repro-safety-generalization-diabetes-testbed`

Rendered logbook:

`https://kondratevakate-repro-safety-generalization-diabe-9febaf4.static.hf.space/`

Hugging Face revision:
`e0e6c83e3ff32d195b9569f7834a1fa4d19a8b5b`.

The Space is public, its runtime reports `RUNNING`, and the Hub, rendered
logbook, and raw evaluator URLs return HTTP 200.

## Claim status

| Claim | Verdict | Prepared |
| --- | --- | ---: |
| C1 unified simulator and benchmark | Partially verified | 1 |
| C2 scoped safety-generalization gap | Partially verified | 1 |
| C3 BA-NODE prediction superiority | Not executed | 0 |
| C4 conditional safety theorem | Verified | 2 |
| C5 released predictive shield | Partially verified | 1 |
| **Prepared** |  | **5/10** |

## Verification

Executed successfully:

```text
python -m pytest ../official/GlucoSim/tests -q
python audit_simulator.py --output results/simulator_execution.json
python audit_theory_and_release.py \
  --glucoalg ../official/GlucoAlg \
  --output results/theory_and_release.json
python audit_generalization_gap.py \
  --checkpoint ../official/models/t1d-adolescent-cpo/checkpoints/seed0/epoch-2441.pt \
  --config ../official/models/t1d-adolescent-cpo/config/seed0/config.json \
  --output results/generalization_gap_cpo_t1d_adolescent_seed0.json
python -m unittest discover -s tests -v
python -m py_compile prepare_official.py audit_simulator.py \
  audit_generalization_gap.py audit_theory_and_release.py build_logbook.py \
  tests/test_audit_theory_and_release.py \
  tests/test_generalization_components.py
python ../../validate_icml_logbook.py \
  --space kondratevakate/repro-safety-generalization-diabetes-testbed
```

Results:

- 22/22 official GlucoSim tests pass;
- 6/6 independent reproduction tests pass;
- all three released environments execute deterministically on CPU;
- the corrected post-step CPO run gives an OOD-minus-ID TIR gap of
  `-10.996172` percentage points and risk-index gap of `+1.986908`;
- 160 hypoglycemia and 160 hyperglycemia theorem implications pass, while
  weakened-reliability mutations fail as expected;
- Trackio logbook validation passes;
- an independent blind preflight returned `APPROVE`.

## Evidence boundary

The C2 metrics use one post-step glucose value per action. The released
evaluator requests `info["cgm"]`, but GlucoSim omits that key and the code
falls back to pre-step glucose. The published result is therefore explicitly
labeled a corrected-protocol mechanism check, not an exact replay of the
released fallback or the paper's full aggregate.

The released shield source contains train/load path, cohort-index, and finite
penalty divergences. These findings do not establish that the paper's clinical
performance claims are false.

## Integrity and privacy

- No leaderboard entries or peer reproductions were inspected or used.
- No patient records, author repositories, paper source, checkpoints,
  credentials, or private medical data are published.
- `diabetes_safety/official/` and generated Trackio working state are ignored.
- Public code and model artifacts are pinned to revisions and the checkpoint
  is verified by SHA-256 before execution.
