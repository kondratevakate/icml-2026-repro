# CAME-Grad publication preflight

Date: 2026-07-29.

## Decision

`PUBLISHED` as an independent mathematical audit at:

`https://huggingface.co/spaces/kondratevakate/repro-came-grad-mathematical-audit`

Rendered logbook:

`https://kondratevakate-repro-came-grad-mathematical-audit.static.hf.space/`

Hugging Face revision:
`f38824206061448bbcd1dbb0a72ca93ba75418d8`.

The Space is public, uses the static SDK, reports `RUNNING`, and the rendered
URL returns HTTP 200 with the expected paper title.

## Forecast calibration

The frozen pre-execution forecast was `6/10` realistic and `7/10` stretch.
Blind preflight rejected an initial `7/10` because the formal definition of
geometric validity applies to the Stage 1 rectified direction, not necessarily
the final fused direction. The corrected, independently approved result is
`6/10`; a strict evaluator floor is `5/10`.

## Claim status

| Claim | Verdict | Prepared |
| --- | --- | ---: |
| C1 two-task interaction identity | VERIFIED FOR EXACT IDENTITY | 2 |
| C2 unconditional Stage 1 Pareto guarantee | FALSIFIED | 2 |
| C3 target norm and scalar covariance identity | PARTIALLY VERIFIED | 1 |
| C4 validity scope after adaptive fusion | QUALIFIED | 1 |
| C5 eight-method clinical efficacy | INCONCLUSIVE, NOT EXECUTED | 0 |

## Verification

Executed successfully:

```text
python prepare_official.py
python audit_claims.py \
  --official-code-root ../official/code \
  --paper-source ../official/source/example_paper.tex \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
python -m py_compile audit_claims.py build_logbook.py prepare_official.py \
  tests/test_audit_claims.py
python ../../validate_icml_logbook.py \
  --space kondratevakate/repro-came-grad-mathematical-audit
```

Results:

- six unit tests pass;
- all C1-C4 paper anchors are present in the pinned TeX source;
- the exact C2 primal and dual both recover `u*=-2.25`, with worst task inner
  product `-2.25`;
- the full C4 calculation starts with positive inner products
  `[2.6645898, 2.3385255]` and ends with
  `[5.7388625, -0.2426039]`;
- Trackio logbook validation passes;
- independent blind preflight returned `APPROVE` with no remaining blockers.

## Integrity and privacy

- No leaderboard entries or peer reproductions are used as evidence.
- No medical images, reports, patient records, author code, paper source,
  checkpoints, weights, cached author metrics, or secrets are published.
- `camegrad/official/` and Python bytecode are excluded by `.gitignore`.
- The public artifact contains the exact evidence JSON and embeds the
  independent audit source on each claim page.
- The clinical claim remains inconclusive because the public author repository
  imports an absent optimizer module and states that the core optimizer and
  weights are temporarily withheld.

## Remaining empirical route

C5 requires the released optimizer or an independently validated
implementation, all eight report-generation baselines, MIMIC-CXR and IU X-Ray
splits, model training or author predictions, and CheXbert evaluation. It is
intentionally deferred rather than reconstructed from paper tables.
