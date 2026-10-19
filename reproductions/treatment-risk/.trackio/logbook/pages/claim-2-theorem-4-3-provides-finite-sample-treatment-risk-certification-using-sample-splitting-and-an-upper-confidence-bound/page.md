# Claim 2: Theorem 4.3 provides finite-sample treatment-risk certification using sample splitting and an upper confidence bound.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a0cf35143015", "created_at": "2026-07-28T16:28:29+00:00", "title": "Claim 2: Theorem 4.3 provides finite-sample treatment-risk certification using sample splitting and an upper confidence bound."}
-->
## Claim

Theorem 4.3 combines policies learned on `D_m`, pointwise upper confidence
bounds on independent `D_n`, and the prefix constraint in Eq. (14) to certify
treatment risk at level `1-alpha`.

```powershell
python audit_theorem43.py --maximum-grid-size 10 --output evidence/theorem43_audit.json
```

## Published proof defect

Equation (18) takes an expectation of the empirical constraint for the policy
learned on the same `D_m` and identifies it with fresh-sample risk. That
exchange is not valid for a data-dependent policy.

An exact two-atom counterexample uses tolerance `tau=1/4` and two policies.
Policy `j` has value `1/4` on atom `j` and `3/4` on the other atom. With one
training atom, Eq. (12) selects the matching policy:

| Quantity | Value |
| --- | ---: |
| Empirical constraint, every training sample | 0.25 |
| Expected empirical value | 0.25 |
| Expected fresh-sample value | 0.50 |
| Excess over tolerance | +0.25 |

Thus even the theorem's empirical equality premise does not justify Eq. (18).

## Certificate engine

The Bentkus UCB in Eq. (15) was independently evaluated by summing every
binomial count for:

```text
n = 5, 8, 10, 20, 50
alpha = 0.05, 0.10, 0.20
p = 0.05, 0.10, ..., 0.90, 0.95
```

All `165/165` instances meet coverage `>= 1-alpha`; the smallest margin is
`+0.018750`. Multiplying the UCB by `0.98` creates three violations, and
replacing it with the raw empirical mean creates `165/165` violations.

## Finite-grid repair

For an ordered finite tolerance grid, unsafe selection implies undercoverage at
the first unsafe candidate: Eq. (14) requires every preceding UCB to be below
`tau`, including that candidate. Exhaustive enumeration through grid size 10
checked `2,046` safety patterns and `1,396,054` UCB pass/fail patterns. There
are zero counterexamples when the first unsafe UCB covers.

The repair needs assumptions not stated in the theorem:

1. the candidate tolerance set is finite and ordered;
2. Eq. (13) holds conditional on the independent learning split `D_m`;
3. the implementation uses the prefix rule exactly;
4. `p_pi(A=1)` and `V_max` are fixed independently of `D_n`.

## Verdict: PARTIAL

The published proof is invalid at Eq. (18), but the implementable finite-grid
certificate is not falsified and admits a separate proof. A full `VERIFIED` or
`FALSIFIED` verdict for the continuum statement would overclaim the evidence.
