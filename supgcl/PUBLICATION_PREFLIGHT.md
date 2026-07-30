# SupGCL publication preflight

Date: 2026-07-30.

## Decision

`READY; PUBLICATION BLOCKED BY HUGGING FACE DAILY SPACE-CREATION LIMIT`.

Target Space:

`https://huggingface.co/spaces/kondratevakate/repro-supervised-graph-contrastive-learning-grn`

The authenticated publish attempt on 2026-07-30 returned HTTP 429:
`You have exceeded the rate limit for space creation (20 per day)`.
The complete upload tree is ready at `supgcl/.trackio/logbook/`; no additional
computation is required after the limit resets.

## Claim status

| Claim | Verdict | Prepared |
| --- | --- | ---: |
| C1 Theorem 1 and LINCS supervision | Verified | 2 |
| C2 Corollary 1 temperature limit | Verified | 2 |
| C3 node-level TCGA performance | Not executed | 0 |
| C4 graph-level TCGA performance | Not executed | 0 |
| C5 superiority across 13 tasks | Not executed | 0 |
| C6 embedding-space analysis | Not executed | 0 |
| **Prepared** |  | **4/12** |

## Verification completed

```text
python prepare_official.py
python audit_claims.py \
  --code-root ../official/code \
  --paper-source ../official/paper \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
python -m py_compile prepare_official.py audit_claims.py build_logbook.py \
  tests/test_audit_claims.py
python ../../validate_icml_logbook.py \
  --space kondratevakate/repro-supervised-graph-contrastive-learning-grn
```

Results:

- pinned code commit and three arXiv artifact hashes pass;
- 100 independent KL decompositions agree to maximum absolute error
  `2.22e-16`;
- 25 independent temperature-limit probes have zero uniform-convergence
  monotonicity failures;
- at `tau_a=100000`, worst total-loss distance is `3.38e-7` and worst
  augmentation KL is `1.35e-10`;
- 5/5 independent tests pass;
- Trackio challenge validation passes.
- an independent blind preflight returned `APPROVE` with no blocking findings
  and confirmed that 4/12 is defensible.

## Integrity

- No leaderboard entries or peer reproductions were inspected or used.
- No TCGA/LINCS records, author repository, paper source, credentials,
  checkpoints, or paper-reported metric caches are published.
- `supgcl/official/` is ignored.
- The public preparation script fetches and verifies pinned primary artifacts.
