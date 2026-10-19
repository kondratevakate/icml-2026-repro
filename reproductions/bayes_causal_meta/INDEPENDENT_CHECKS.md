# Independent checks

## Prepared score: 5 / 10

| Claim | Score | Finding |
|---|---:|---|
| C1 Embedding-conditional prior | 1/2 | The release adds an embedding-dependent offset to global parameters, but normalizes every non-zero `Wz` to a fixed fraction of the global norm. It is scale-invariant away from zero and discontinuous at zero, unlike the paper's linear `theta + Wz`. |
| C2 Prior-risk continuity and decomposition | 2/2 | The Gaussian TV/Pinsker bound and the three-term triangle decomposition check out for the paper's stated linear prior. |
| C3 Negative-transfer theorem | 1/2 | A condition-number-aware constant can support the argument. Appendix A.2 instead drops the condition number and sets `C=kappa_0/kappa`; a 2D diagonal counterexample satisfies this stated embedding condition while failing the parameter-space condition needed by the proof. |
| C4 Synthetic task-shift evidence | 1/2 | The released adaptive toy training and BALD paths execute on CPU after two packaging/console workarounds. The smoke is not a canonical 30-run reproduction, and the training seed is incomplete. |
| C5 Clinical cross-disease evidence | 0/2 | UK Biobank data, clinical outputs, and checkpoints are absent. |

## Released prior versus paper prior

For global parameters `[3,4]`, `adaptation_scale=0.2`, and a one-dimensional
embedding map with `Wz=[z,0]`, the official code returns:

- `z=0`: offset `[0,0]`;
- `z=1e-9`: offset `[1,0]`;
- `z=1`: offset `[1,0]`;
- `z=2`: offset `[1,0]`.

Thus the released map has a unit jump at zero and exactly zero scale response
between `z=1` and `z=2`. With unit Gaussian covariance, the exact TV jump is
`0.382925`, while the paper's linear Lipschitz right-hand side at
`epsilon=1e-9` is `5e-10`.

This does not refute Proposition 4 for the paper's explicitly stated linear
Gaussian prior. It shows that the proposition does not apply to the released
normalized prior.

## Proposition 4

For equal-covariance one-dimensional Gaussians, exact total variation was
checked at embedding distances `0`, `0.01`, `0.1`, `1`, and `3`. Every value
was below the paper's Pinsker bound `distance/2` for `M=||W||=sigma=1`. The
expert/causal/OOD decomposition is the triangle inequality through
`z_hat`, `z_tilde`, `z_true`, and the source mean.

## Theorem 5 constant

Let `W=diag(10,1)`, `kappa=1`, `kappa_0=0.5`, target embedding `z=[0,1]`,
and embedding error `e=[0.1,0]`. The appendix's condition holds:

`||e||=0.1 <= (kappa_0/kappa)||z||=0.5`.

But the parameter-space condition required one line earlier fails:

`kappa||We||=1 > kappa_0||Wz||=0.5`.

A sufficient norm-only constant needs the singular-value ratio:
`C <= kappa_0 / (kappa * cond(W))`, which is `0.05` here.

## Toy smoke

The official sequence/adaptive path was run for one epoch, one MC sample, and
one inner update on CPU with `--random_seed 999`.

- attempt 1: validation AUROC `0.7188`; failed because the entrypoint did not
  create the parent `results/` directory;
- attempt 2: validation AUROC `0.7276`, average test AUROC `0.6160`; metrics
  were saved, then Windows cp1252 failed on the final emoji print;
- attempt 3 with `PYTHONUTF8=1`: validation AUROC `0.6531`, average test AUROC
  `0.56660`, and average test AUPRC `0.38805`; completed.

The same declared seed produced a validation-AUROC range of `0.0745`.
`method/main.py` does not call `torch.manual_seed` or `numpy.random.seed`; the
argument controls data splits but not model training.

A separate reduced BALD smoke completed for all five target tasks with two
queries per target and 20 SVI steps per query.

## Release inventory

- official commit: `29037c2193e72b286cef8ec2f5dc139a18aea38a`;
- 38 tracked files and 16 Python files, all compiling;
- example tabular rows: 12,500;
- example embeddings: 25 tasks;
- no tests, cached metrics, checkpoints, or UK Biobank data.

## Visual PDF check

Pages 6, 7, 8, 11, 12, and 13 of arXiv v1 were rendered and visually checked.
The propositions, clinical table, assumptions, and Appendix Theorem 5 proof
are legible and agree with the TeX evidence used by this audit.

