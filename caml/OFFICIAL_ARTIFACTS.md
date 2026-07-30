# Official artifacts

## Paper and source

- OpenReview: `Fisw2kc7EY`
- arXiv: `2605.25001v1`
- PDF SHA-256:
  `4CA6809002A0AB786781B53E30DC7C200C7A1BF7B9ACEE676D1CF09C8AF6152C`
- source archive SHA-256:
  `13C19394C76C86F337F8FE37DD538134FBFD93C5B8B6AEE4D64EFC6CDEDD1646`

## Official code

- repository: `YichenLuo-0/CAML`
- audited commit: `8f4daabf0db60f4f185a1a9b791ad7d7be41a033`
- tracked files: 28
- analytic entrypoints: Heat, Poisson, Navier-Stokes, Helmholtz
- backbones: MLP, PirateNet, PINNsFormer
- released loss implementations: standard PINN and CAML

There are no cached predictions, metric files, checkpoints, tests, environment
lockfile, or requirements file. All empirical values require fresh training.
The README says no dependency beyond standard PyTorch, but benchmark modules
also require the unlisted `overrides` package.

All tasks use manufactured or analytic PDE solutions. No external dataset is
needed.
