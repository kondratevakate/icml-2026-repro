# Claim decomposition

## C1 - Loss-valley theorem (2 points)

Audit whether non-uniqueness of a PDE residual equation is sufficient for a
flat, connected, non-isolated solution set in function and parameter space.

## C2 - Aligned-constraint mechanics (2 points)

Check the linear closed-form offset, nonlinear Newton update, reset semantics,
and delay/ramp schedule against the printed equations.

## C3 - Heat benchmark (2 points)

Reproduce the five-seed MLP Heat result and compare CAML with the released
standard PINN path at matched sampling and training budgets.

## C4 - Cross-PDE MLP results (2 points)

Exercise Heat, Poisson, Navier-Stokes, and Helmholtz entrypoints and compare
their aggregate accuracy/convergence behavior with Table 2.

## C5 - Backbone generality (2 points)

Check CAML with the MLP, PirateNet, and PINNsFormer implementations across the
paper's reported benchmark matrix.

## C6 - Components and limitations (2 points)

Audit aligned-constraint and delay-residual ablations, schedule sensitivity,
the benign toy-Poisson limitation, and reported overhead.
