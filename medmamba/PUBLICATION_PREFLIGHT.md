# MedMamba publication preflight

Date: 2026-07-29.

## Decision

`PUBLISHED` as an implementation-conformance reproduction at:

`https://huggingface.co/spaces/kondratevakate/repro-medmamba-implementation`

Rendered logbook:

`https://kondratevakate-repro-medmamba-implementation.static.hf.space/`

Hugging Face revision:
`77f213a24cd4ea2d6487f745a1f98cba26a26416`.

The Space is public, its runtime reports `RUNNING`, and the rendered URL
returns HTTP 200 with the expected paper title.

## Forecast calibration

The frozen pre-execution forecast was `8/12` realistic and `10/12` stretch.
The completed evidence supports `10/12`: five falsified implementation claims
and one inconclusive empirical claim.

## Claim status

| Claim | Verdict | Prepared |
| --- | --- | ---: |
| C1 channel-wise MCE | FALSIFIED IN RELEASED IMPLEMENTATION | 2 |
| C2 zero-padded difference | FALSIFIED IN RELEASED IMPLEMENTATION | 2 |
| C3 frequency-specific filter | FALSIFIED IN RELEASED IMPLEMENTATION | 2 |
| C4 sample-conditioned graph | FALSIFIED IN RELEASED IMPLEMENTATION | 2 |
| C5 graph diffusion and channel scan | FALSIFIED IN RELEASED IMPLEMENTATION | 2 |
| C6 five-dataset empirical results | INCONCLUSIVE, NOT EXECUTED | 0 |

## Verification

Executed successfully:

```text
python audit_claims.py \
  --official-code-root ../official/code \
  --paper-source-root ../official/source \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
python -m py_compile audit_claims.py build_logbook.py tests/test_audit_claims.py
python ../../validate_icml_logbook.py \
  --space kondratevakate/repro-medmamba-implementation
```

Results:

- six unit tests pass;
- Trackio logbook validation passes without warnings;
- all paper anchors are present in the pinned arXiv source;
- adjacency changes by `0.5020374` after graph-parameter mutation while block
  output changes by exactly zero;
- classification-output backpropagation leaves all graph gradients `None`;
- all five traced Mamba calls use time length 16 as their sequence axis, not
  the four-channel axis.

The Torch `padding='same'` warning is emitted by the released spatial Conv1d
for an even kernel. It does not affect any asserted result.

## Integrity and privacy

- No leaderboard entries or peer reproductions were inspected or used.
- No medical data, patient records, author code, paper source, checkpoints, or
  cached author metrics are included in the publishable bundle.
- `medmamba/official/` is excluded by `.gitignore`.
- The CPU identity Mamba substitute is used only for shape and dependency
  tracing, never as evidence about accuracy or Mamba numerical behavior.
- Machine-readable evidence records the pinned author commit and SHA-256
  hashes for every audited author-code file and paper-source file.

## Remaining empirical route

C6 requires all five subject-independent datasets, the declared subject
splits, seeds 41-45, missing dataset configurations, baselines, component
ablations, graph ablations, and robustness experiments. The public repository
only supplies an APAVA shell script and no checkpoints or cached metrics, so
this route is intentionally deferred to a GPU server after data preparation.

