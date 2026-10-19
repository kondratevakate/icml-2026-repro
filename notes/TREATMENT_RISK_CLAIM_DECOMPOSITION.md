# Treatment Risk Claim Decomposition

Paper: *Learning Treatment Allocations with Risk Control Under Partial
Identifiability* (`9kWVQtpvJi`, arXiv `2505.08378v1`).
Source snapshot SHA-256:
`AFDEAA8C905A68EB5F83CC32147A6D8FDF90651ED2D32521711F5829241D7028`.
Date: 2026-07-28.

## Decision

Proceed as the next local medical paper. The honest target is `4/10` from
Lemma 4.1 and the synthetic experiment, with `6/10` possible if Theorem 4.3
survives an adversarial proof audit or yields a finite counterexample.

| Claim | Local route | Target | Stop condition |
| --- | --- | ---: | --- |
| C1, Lemma 4.1 risk bounds | Independently enumerate finite `(X,U,A,L)` laws, compute exact policy risks and Eqs. (9)-(11), verify equality at `Gamma=1`, and underestimate `Gamma` as a destructive control | 2 | Stop if the selection/propensity model cannot be represented exactly |
| C2, Theorem 4.3 certificate | Audit Eqs. (16)-(20), then prove or refute the data-dependent-policy step with exhaustive finite policy classes | 2 | Do not claim from a Bentkus implementation alone |
| C3, synthetic experiment | Reproduce `m=n=1000`, 200 tolerance values, one-split trees, 1000 datasets, `alpha=0.1`, and both `Gamma=1,2` confounding regimes | 2 | Stop if the exact greedy policy search cannot be reconstructed |
| C4, Tennessee STAR | Hold until the exact cohort and preprocessing are verified | 0 initially | No substitute education dataset |
| C5, International Stroke Trial | Hold until exact 18,298-row data and 23-covariate preprocessing are verified | 0 initially | No MIMIC proxy |

## Theorem 4.3 Proof Risk

The proof uses the empirical constraint on the policy learned from `D_m`,

```text
mean_m V_i(t) <= t,
```

and then states in Eq. (18) that taking expectations gives

```text
t >= (1/m) sum_i E[V_i(t)] = E[V(t)] = T(t).
```

The last equality is not automatic because `pi(X;t)` is selected using the
same `D_m` containing the `V_i(t)` terms. In-sample loss of a data-dependent
policy need not have the same expectation as its loss on a fresh observation.
No uniform-convergence, stability, finite-class, or independent training split
assumption is stated at this step.

This is currently a **proof gap, not a falsification**. Before assigning a
verdict:

1. Formalize the smallest finite policy class where Eq. (12) selects a
   data-dependent policy.
2. Enumerate all training and certification samples exactly.
3. Require the theorem's equality premise at `t=tau`.
4. Check whether `P(T(pi)>tau)` can exceed `alpha`.
5. If no counterexample exists, identify the hidden monotonicity or
   exchangeability argument that repairs Eq. (18).

## Synthetic Protocol

Main-text Section 5.1 specifies:

- `X1,X2 ~ Uniform(30,80)`;
- nominal propensity `sigmoid(0.5-(X1-30)/50)`;
- untreated loss probability `0.8`;
- treated loss probability `0.01*(X1-30)`;
- `D_m=1000`, `D_n=1000`;
- 200 equally spaced nominal tolerances in `(0,0.5]`;
- one-split fast-and-frugal trees with 200 covariate bins;
- miscoverage `alpha=0.1`;
- 1000 independently learned policies.

The confounded case adds `U~Uniform(0,1)`, true odds distortion `Gamma=2`,
and Eq. (24)'s treated loss. The deciding result must show:

- valid 90% treatment-risk coverage when the assumed `Gamma` is valid;
- deliberate undercoverage when `Gamma=1` is used on the `Gamma=2` process;
- conservative but valid behavior when `Gamma=2` is used without confounding.

## Preflight Estimate

| Stage | Time | Expected points |
| --- | ---: | ---: |
| C1 exact finite enumeration | 1-2 h | 2 |
| C2 proof-gap counterexample/search | 2-4 h | 0 or 2 |
| C3 full 1000-dataset simulation | 2-3 h | 2 |
| Logbook, mutation tests, final preflight | 1 h | - |

Conservative total: `4 points / 6-8 hours`. Upside: `6 points` if C2 receives
a decisive verified or falsified verdict.

## Execution outcome

Completed locally on 2026-07-28:

| Claim | Outcome | Evidence |
| --- | --- | --- |
| C1 | `VERIFIED` | `268,800` exact checks; equality exact at `Gamma=1`; both underestimated-`Gamma` controls fail |
| C2 | `PARTIAL` | Eq. (18) counterexample; Bentkus `165/165`; finite-grid first-unsafe repair has zero exhaustive counterexamples |
| C3 | `VERIFIED` | Full `1000 x 4` run; valid arms `8/8`, invalid arm `5/8`, minimum invalid coverage `0.257` |
| C4 | `NOT ATTEMPTED` | Exact STAR cohort unavailable |
| C5 | `NOT ATTEMPTED` | Exact IST cohort unavailable |

Final local forecast is `4-5/10`, not the earlier six-point upside. The
published proof step is defective, but the finite-grid implementation remains
conditionally certifiable, so C2 is neither fully verified nor falsified.
