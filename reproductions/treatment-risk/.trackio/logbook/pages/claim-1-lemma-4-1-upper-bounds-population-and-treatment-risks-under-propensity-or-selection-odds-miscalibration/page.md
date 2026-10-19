# Claim 1: Lemma 4.1 upper bounds population and treatment risks under propensity or selection odds miscalibration.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1128e6ade0d3", "created_at": "2026-07-28T16:28:29+00:00", "title": "Claim 1: Lemma 4.1 upper bounds population and treatment risks under propensity or selection odds miscalibration."}
-->
## Claim

Lemma 4.1 states that the population and treatment risks are upper bounded by
the weighted expectations in Eqs. (9)-(11). For observational data,

```text
W_Gamma = [1 + Gamma_a * (1 / p_hat(A|X) - 1)] * 1(A = pi(X)).
```

The randomized-trial weight instead bounds selection-odds distortion. Both
bounds are claimed to become equalities at `Gamma=1`.

## Independent audit

`audit_lemma41.py` uses exact `fractions.Fraction` arithmetic. It enumerates
binary `(X,U,A,L)` models, all four deterministic policies on binary `X`, and
all 256 deterministic potential-loss tables. Seventy-five assignment/selection
models are checked in each regime, spanning `Gamma in {1, 3/2, 2, 3}` and
boundary/interior odds-distortion patterns.

```powershell
python audit_lemma41.py --output evidence/lemma41_finite_audit.json
```

| Regime | Population-risk checks | Treatment-risk checks | Max excess over bound | `Gamma=1` equality error |
| --- | ---: | ---: | ---: | ---: |
| Observational, Eq. (10) | 76,800 | 57,600 | 0 | 0 |
| Randomized trial, Eq. (11) | 76,800 | 57,600 | 0 | 0 |

The total is `268,800` exact checks. All pass.

## Destructive controls

The data generators use true odds distortion `Gamma=2`, but the audit
deliberately evaluates weights with `Gamma=1`. This violates the lemma's
sensitivity assumption and must break the bound.

| Regime | Population-risk excess | Treatment-risk excess | Detected |
| --- | ---: | ---: | --- |
| Observational | +0.240000 | +0.400000 | yes |
| Randomized trial | +0.241935 | +0.500000 | yes |

This is a direct finite-state audit, not a downstream simulation proxy. It does
not replace a symbolic proof over arbitrary measurable spaces, but it exercises
both published weight formulas, their equality case, and assumption failure.

## Verdict: VERIFIED

No finite counterexample was found, equality at `Gamma=1` is exact, and both
negative controls fail in the predicted direction.
