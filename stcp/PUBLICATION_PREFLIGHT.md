# StCP publication preflight

Date: 2026-07-29.

## Decision

`PUBLISHED` as a conservative `6/12` reproduction logbook at:

`kondratevakate/repro-stable-localized-conformal-prediction-via-transduction`

Hub URL:

`https://huggingface.co/spaces/kondratevakate/repro-stable-localized-conformal-prediction-via-transduction`

Published Space revision:
`a9ac821d4adb05be4f8f6f9bcdd4c54f695de05a`.
The Hub runtime reported `RUNNING` after publication.

The frozen pre-audit forecast was `8/12` realistic and `10/12` stretch.
Current evidence supports six points before fresh canonical experiments.

## Claim status

| Claim | Verdict | Prepared |
| --- | --- | ---: |
| C1 set-stability criterion | VERIFIED | 2 |
| C2 Theorem 4.2 | NOT ESTABLISHED AS WRITTEN | 2 |
| C3 Theorem 4.6 | NOT ESTABLISHED UNDER STATED ASSUMPTIONS | 0 |
| C4 Theorem 4.7 | FALSIFIED AS WRITTEN | 2 |
| C5 Derma/Tissue gains | PENDING FRESH MEDICAL RUN | 0 |
| C6 LogAbs gains | PENDING CANONICAL RUN | 0 |

C2 is supported by a counterexample to the exact proof step, not by a claim
that every corrected theorem is false. C3 is deliberately assigned zero:
the audit found an assumption and moment-control gap but not a full theorem
counterexample.

## Verification

Executed successfully:

```text
python audit_claims.py --source ../official/source/main.tex \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
python -m py_compile audit_claims.py run_official_synthetic.py build_logbook.py
python ../validate_icml_logbook.py \
  --space kondratevakate/repro-stable-localized-conformal-prediction-via-transduction
```

Results:

- five unit tests pass;
- Trackio logbook validation passes without warnings;
- official LogAbs path completes a reduced smoke run at pinned commit
  `84118ddf19efe211250a5f024c1be5a3c8f47b4b`;
- smoke output is explicitly labeled `NO CLAIM VERDICT`;
- the C4 exact counterexample gives `27/31 = 0.8709677 < 0.88`;
- 8,960 lower-endpoint arithmetic failures were found on the declared finite
  `(n,p)` audit grid.

## Integrity and privacy

- No leaderboard entries or peer reproductions were used.
- No medical images, patient data, model weights, author code, paper source,
  or cached author results are included in the publishable bundle.
- `stcp/official/` is excluded by `.gitignore`.
- All publishable files are below 110 KB.
- Machine-readable evidence records source and code SHA-256 values.
- The author repository and paper source are used only for provenance, claim
  anchoring, and the separately labeled official execution path.

## Remaining route to 8/12

Run the canonical LogAbs protocol on a persistent GCP instance:

```text
python run_official_synthetic.py \
  --official-code-root ../official/code \
  --repeats 50 \
  --test-points 500 \
  --epochs 200 \
  --source-n 2000 \
  --n 30 \
  --m 500 \
  --n-grid 50 \
  --output evidence/logabs_canonical.json
```

Only that canonical result is eligible for a full C6 verdict. The medical C5
route remains optional stretch work after the synthetic run.
