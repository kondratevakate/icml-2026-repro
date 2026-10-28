# MNAR OPE Publication Preflight

Date: 2026-07-29.

## Target

- Space:
  `kondratevakate/repro-off-policy-evaluation-for-missingness-aware-policies-in-mdps-with-rewards-missing-not-at-r`
- Paper tag: `paper-vpSFJoxyDz`
- arXiv metadata: `2606.20206`
- Expected judge score: `8/10`
- Prepared verdict total: `10/10`

## Frozen verdicts

| Claim | Verdict | Full evidence |
| --- | --- | --- |
| C1 | `FALSIFIED AS WRITTEN` | Complete wrapped-Gaussian inverse problem with divergent Picard sum. |
| C2 | `VERIFIED` | Source equation, official DGP dependency, and observed-subset residual association. |
| C3 | `VERIFIED` | Theorem audit and 2,000 finite inverse-problem reductions with zero violations. |
| C4 | `VERIFIED` | Full proof ledger, exact recurrence unrolling, bounded concentrability product, and rate substitution. |
| C5 | `VERIFIED` | Source equation, AST dependency audit, and nested logistic diagnostic. |

## Checks

- [x] arXiv source and PDF hashes frozen.
- [x] official repository pinned to
  `4231ba5d46046c66c0efae6d58662bfca5087147`.
- [x] official CPU smoke path ran all five estimators.
- [x] independent audit emits all five prepared verdicts.
- [x] five unit tests pass.
- [x] Trackio pages use short Windows-safe slugs.
- [x] every claim page contains the verbatim anchored claim.
- [x] executive summary and poster are pinned.
- [x] MIMIC-III cached outputs are explicitly excluded as independent evidence.
- [x] local `validate_icml_logbook.py` passes for the exact target Space.
- [x] reproducibility bundle hashes are frozen in
  `repro_mnar_ope/INTEGRITY_MANIFEST.sha256`.

## Known limitations

- C3 accepts the cited projected empirical-process result under the theorem's
  explicit assumptions and independently verifies its L2 reduction.
- C4 reports two appendix bookkeeping defects: a temporarily dropped summation
  sign and an implicit `zeta/T` union-bound adjustment. The subsequent display
  and final logarithmic factor restore the intended bound.
- The C1 verdict targets the shortened challenge claim. The qualified paper
  theorem includes the missing Picard regularity condition and is not
  falsified by the counterexample.

## Published state

- Hub Space:
  `https://huggingface.co/spaces/kondratevakate/repro-off-policy-evaluation-for-missingness-aware-policies-in-mdps-with-rewards-missing-not-at-r`
- Canonical static host:
  `https://kondratevakate-repro-off-policy-evaluation-for-m-acf261e.static.hf.space`
- Remote commit:
  `3b7667b50e955886c7b17aff4a8e18b43b401c3e`
- Runtime: `RUNNING`, static SDK.
- Remote paper tags: `paper-vpSFJoxyDz`, `arxiv:2606.20206`.
- Remote `index.md`, executive summary, five claim pages, and conclusion all
  match their local SHA-256 hashes.
