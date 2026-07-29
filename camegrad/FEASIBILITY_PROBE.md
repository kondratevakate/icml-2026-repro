# CAME-Grad feasibility probe

Paper: *The Double Dilemma in Multi-Task Radiology Report Generation: A
Gradient Dynamics Analysis and Solution* (`T7y2wavrFM`, arXiv `2605.22635v2`).

## Provenance

- Author repository: `https://github.com/vpsg-research/CAME-Grad`
- Pinned commit: `79059e39060d13ef6b6cb2ea9ad7a5d8e2519c83`
- arXiv source archive SHA-256:
  `9f5109704136930a9453e0995aad73ec6f794f51ff15c55e54a3f031369199d1`

No peer reproduction is used as evidence. A broad local search accidentally
surfaced an old peer-index row for this paper; it is excluded from all
forecasting, claim selection, and verdicts.

## Released execution surface

The repository contains a radiology-report training skeleton and exposes a
`--use_came_grad` flag. However, `modules/trainer.py` imports
`modules.CAME_Grad`, while `modules/CAME_Grad.py` is absent. The README
explicitly states that the core optimizer source and model weights are
temporarily withheld.

Consequently, the advertised CAME-Grad training command cannot import, and the
MIMIC-CXR/IU X-Ray empirical tables cannot be rerun from the public artifact.
The paper source does fully specify the three mathematical stages, enabling an
independent CPU audit.

## Frozen forecast

Frozen before executable counterexamples:

- realistic: `6/10`;
- stretch: `7/10`;
- local effort: 2-4 hours;
- clinical empirical claim: `0/2` without weights, predictions, and optimizer
  source.

## Decision

`GO` for an independent mathematical/algorithmic audit.

Stop conditions:

- do not reconstruct clinical metrics from paper tables;
- do not present an independent implementation as the withheld author code;
- distinguish the exact two-task interaction identity from the paper's
  approximation that drops inter-auxiliary terms;
- treat numerical epsilon effects as qualifications unless they invalidate an
  explicit exact statement;
- require explicit counterexamples before falsifying geometric validity.
