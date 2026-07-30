# Independent checks

## Outcome

Prepared score: **4/12**, below the frozen forecast of **7/12**.

## C1 - Unsupported (0/2)

The headline theorem is false under its stated premise. Consider the
zero-dimensional nonlinear operator `N(u) = u^2` with `f = 1`. It has two
solutions, `{-1, +1}`, but both are isolated and the solution set is
disconnected. Thus non-uniqueness alone does not imply a connected,
non-isolated valley.

The appendix proof silently strengthens the nonlinear case by assuming a
non-trivial kernel of `DN[u*]`, a constant-rank/regular-value condition, and a
continuous solution-family parameterization. Mapping the family into network
parameter space additionally requires explicit representability and
regularity assumptions. Those conditions do not follow from the main
theorem's sole non-uniqueness premise.

## C2 - Verified (2/2)

On a deterministic synthetic linear residual/BC system, the official
closed-form offset equals an independent implementation exactly (absolute
difference `0.0`). The offset is detached as described, and the delay/ramp
schedule matches the printed piecewise definition at every boundary.

The implementation has an uncovered edge case: if every zeroth-order
coefficient is zero, the closed-form denominator is zero and the official
code returns `NaN`. The paper restricts the aligned construction to
zeroth-order terms, but neither code nor documentation guards this case.

## C3 - Partial (1/2)

The official Heat path imports after installing the undocumented `overrides`
package. A 10-epoch, seed-999 CPU smoke run completes and produces finite
loss/error values. It is deliberately not treated as Table-2 reproduction.

The released Heat settings do not match the paper:

| Parameter | Paper MLP | Code |
| --- | ---: | ---: |
| boundary weight | 5 | 1 |
| stop threshold | 2e-3 | 1e-3 |

Ten canonical-size Heat epochs took about 2.3 seconds after warm-up on CPU;
five runs of at least 6,000 epochs plus baselines are outside a reasonable
local-hour audit.

## C4 - Unsupported (0/2)

The four entrypoints compile, but five code/paper configuration mismatches
prevent exact table execution:

- Heat: boundary weight and stop threshold;
- Navier-Stokes: stop threshold `1e-2` in code versus `5e-3` in paper;
- Helmholtz: boundary weight `1` versus `10`, and `T_min=6000` versus `4000`.

No cached outputs are released, so there is no official numerical artifact
against which to adjudicate the settings.

## C5 - Unsupported (0/2)

Three backbone classes are present, but each entrypoint is hardcoded to MLP.
Switching the string does not apply the backbone-specific learning rates,
weights, budgets, or thresholds printed in the appendix. Four of the six
paper baseline loss functions are also absent.

## C6 - Partial (1/2)

The independent tests verify the two core component mechanics: the aligned
offset and delay/ramp gate. The paper's ablation and benign-case claims cannot
be regenerated directly because no ablation, optimizer-comparison, schedule
sensitivity, or overhead runner is released.
