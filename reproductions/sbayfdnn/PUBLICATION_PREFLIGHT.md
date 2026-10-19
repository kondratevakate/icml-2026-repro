# sBayFDNN publication preflight

Date: 2026-07-30.

## Decision

`PUBLISHED` as an independent method and theorem audit at:

`https://huggingface.co/spaces/kondratevakate/repro-sparse-bayesian-functional-deep-learning`

Rendered logbook:

`https://kondratevakate-repro-sparse-bayesian-functional-b876871.static.hf.space/`

Hugging Face revision:
`f6906e055b71d349f21f89a0d532c8a220eb55a7`.

The Space is public, its runtime reports `RUNNING`, all challenge tags are
present, and the Hub, rendered logbook, and raw evaluator URLs return HTTP 200.

## Claim status

| Claim | Verdict | Prepared |
| --- | --- | ---: |
| C1 spike-and-slab region selection | Verified | 2 |
| C2 Theorem 5.4 approximation bound | Falsified as stated | 2 |
| C3 Theorem 5.7 posterior contraction | Inconclusive proof gap | 0 |
| C4 Theorem 5.9 selection consistency | Inconclusive | 0 |
| C5 ECG performance | Not executed | 0 |
| C6 Tecator performance | Not executed | 0 |
| **Prepared** |  | **4/12** |

## Verification

Executed successfully:

```text
python prepare_official.py
python audit_claims.py \
  --code-root ../official/code \
  --paper-source ../official/paper/ICML-FDNN.tex \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
python -m py_compile prepare_official.py audit_claims.py build_logbook.py \
  tests/test_audit_claims.py
python ../../validate_icml_logbook.py \
  --space kondratevakate/repro-sparse-bayesian-functional-deep-learning
```

Results:

- the pinned author code and arXiv source pass revision and SHA-256 checks;
- 1,201 independent Bayes-formula points agree with the released PIP to
  maximum absolute error `7.22e-15`;
- the independent basis-to-region mapping agrees exactly;
- the positive `E_n=1/n` Theorem 5.4 counterexample satisfies all three
  stated assumptions and its error/rate ratio grows from `8.91` to `319.00`;
- Theorem 5.7 is conservatively labeled a proof gap, not falsified;
- 5/5 independent tests pass;
- Trackio logbook validation passes;
- an independent blind preflight returned `APPROVE`.

## Integrity and privacy

- No leaderboard entries or peer reproductions were inspected or used.
- No ECG records, author repository, paper source, cached author metrics,
  credentials, or private medical data are published.
- `sbayfdnn/official/` is ignored.
- The published preparation script fetches pinned public artifacts and
  verifies the code revision, source-archive SHA-256, and TeX SHA-256 before
  execution.
