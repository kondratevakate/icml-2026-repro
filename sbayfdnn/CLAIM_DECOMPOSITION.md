# Claim decomposition: sBayFDNN

## Frozen scorecard

| Claim | Reproduction route | Planned verdict | Prepared |
| --- | --- | --- | ---: |
| C1 spike-and-slab PIPs induce spline-region selection | Independently derive the group-normal Bayes formula, compare it over a numerical grid with the released implementation, and audit basis-to-interval mapping | Verify if formula and mapping agree | 2 |
| C2 Theorem 5.4 approximation bound | Audit all stated assumptions and construct a bounded-class counterexample | Falsified as stated because `E_n` is unconstrained | 2 |
| C3 Theorem 5.7 posterior contraction | Compare the theorem's rate condition with every proof use | Inconclusive proof gap; no failed posterior sequence established | 0 |
| C4 Theorem 5.9 region selection consistency | Audit posterior-to-MAP plug-in transition and stated assumptions | Inconclusive unless the cited transition is independently established | 0 |
| C5 ECG performance and QRS alignment | Requires released preprocessing, splits, baselines, and repeated full fits | Not executed | 0 |
| C6 Tecator performance and spectral interpretation | Requires released preprocessing, splits, baselines, and repeated full fits | Not executed | 0 |
| **Prepared** |  |  | **4/12** |

## Evidence boundaries

- Cached notebook outputs are not reproduction evidence.
- A compact synthetic run can validate executable method mechanics but cannot
  replace the paper's multi-scenario, 50-replication comparisons.
- C2 concerns the theorem exactly as quantified over its unconstrained
  parameter-bound sequence. Adding a sufficiently large lower bound on `E_n`
  repairs this counterexample but changes the statement.
- C3 is only a proof gap. The reversed rate condition is not by itself a
  counterexample to posterior contraction.
- C4 receives no credit from a proof that introduces an additional
  posterior-to-MAP equivalence condition not present in the theorem statement.
- C5 and C6 remain zero until the exact real-data protocols are released or
  independently reconstructed.
