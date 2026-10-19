# ROCP proof-backed claim audit

Paper: *Optimal Decision-Making Based on Prediction Sets*,
arXiv `2602.00989v3`, OpenReview `VAXW59dyfk`.

## C1: Lemma 2.1 and Theorem 2.2

For a fixed action, let `p = Q(S)`, let `l_in` be its largest loss in `S`,
and let `l_out` be its largest loss outside `S`. Every feasible adversarial
distribution satisfies

```text
E_Q[l(a,Y)] <= p*l_in + (1-p)*l_out,  p >= 1-alpha.
```

If `l_out > l_in`, the right-hand side is maximized at `p=1-alpha` and is
attained by placing masses `1-alpha` and `alpha` on in-set and out-of-set
maximizers. If `l_out <= l_in`, it is maximized at `p=1` and attained by an
in-set maximizer. Thus the primal value is exactly

```text
l_in + alpha * max(l_out - l_in, 0).
```

Minimizing this value over actions gives Equation (4). In the finite-X
specialization of Equation (5), an adversary can concentrate all feature mass
on the context with the largest minimized value, so the outer value is the
maximum over contexts.

`audit_claim1.py` independently solves the defining adversarial problem as a
linear program. It checks the formula and the complete set of minimizing
actions over an exhaustive 2-label/2-action integer-loss family and larger
seeded cases. It also records an asymmetric-loss counterexample where the
in-set max-min action has value `1` but robust value `5.95`, while the ROCP
action has robust value `2`.

## C3: Algorithm 1 finite-sample coverage

For the true candidate label of held-out point `j`, Algorithm 1 augments the
`n` calibration observations with that candidate. Across all possible held-out
indices, the augmented sample is the same exchangeability orbit of `n+1`
labeled observations. The constrained beta is therefore the same deterministic
function of that orbit. Its feasibility constraint ensures that at least
`ceil((n+1)*(1-alpha))` positions are covered. A uniformly held-out position
then has conditional coverage at least `1-alpha`. Any finite exchangeable law
is a mixture over such permutation orbits, preserving the bound.

`audit_claim3_exchangeability.py` checks this argument exhaustively over every
multiset orbit from a nine-observation support, several sample sizes, and
multiple alpha values. The implementation enumerates every finite selector
breakpoint and does not assume empirical coverage is monotone in beta.

## Independent implementation findings

The released `rocp.py` is useful for the empirical experiments, but differs
from the paper in two places relevant to C3:

1. `calibrate_beta()` omits the candidate test point, while Algorithm 1 includes
   it in the `(n+1)` constraint separately for every candidate label.
2. At `t=0`, Remark 3.2 sets `theta(x,0)=M(a(x,0))` and `C(x,0)=Y`. The released
   `compute_theta_and_action()` instead uses the minimum action loss as theta,
   which can produce a strict subset of `Y`.

These findings establish a code-to-paper divergence. They do not falsify the
paper's finite-sample theorem.
