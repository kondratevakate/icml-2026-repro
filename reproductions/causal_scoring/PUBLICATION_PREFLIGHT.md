# Tailored scoring rules publication preflight

Date: 2026-07-29

## Decision

`PUBLISHED` after independent blind preflight approval.

Prepared score: **7/12**.

Space:
`https://huggingface.co/spaces/kondratevakate/repro-tailored-scoring-rules-causal-inference`

Rendered logbook:
`https://kondratevakate-repro-tailored-scoring-rules-caus-c220886.static.hf.space`

Hugging Face revision:
`903f04506beb6966ebf8efc4db53e3be30ee6294`.

The public static Space reports `RUNNING`. The Hub page, rendered logbook, and
raw runnable script return HTTP 200 and contain the expected paper title.

## Frozen artifacts

- arXiv source archive SHA-256:
  `7862c54d7da9cabecfefbc7bd2308c7cdff507223be68ed9c9ebaada73fe1129`;
- paper PDF SHA-256:
  `adc0defaed5b133c8128b1ee3cb0962e07e6ab828700817855a3a6876f4fccaa`;
- main TeX SHA-256:
  `52494bc18300fe22d13650119ec5919323f09c9ce04ac3c2e0a8925c861db6d5`;
- seed-panel SHA-256:
  `b9727908367ea1475e75f3122df53b9b984a131bc7a2817e6beec661ec20c37b`.

## Verification

- Six deterministic unit and mutation tests pass.
- A fresh audit reproduces the frozen evidence and prepared score.
- The challenge logbook validator passes.
- The public bundle contains the audit script, tests, requirements, claims,
  evidence, and official-artifact preparation script.
- The independent blind preflight initially blocked over missing runnable code
  and three overbroad phrases. All four findings were corrected; the repeat
  review approved publication and retained the 7/12 score.

## Evidence boundary

- C1 scores 1/2: the theorem-specific pointwise bias bound is supported, while
  ordinary K-fold cross-fitting does not establish unconditional independence.
- C4 scores 1/2: the canonical construction is verified; only the asymptotic
  no-vanishing interpretation is rejected.
- C5 scores 1/2: the result is a scoped independent Kang-Schafer simulation
  with an explicit noise assumption and fixed regularization, not an exact
  paper-table reconstruction.
- C6 scores 0/2: IHDP, Jobs, and ACIC were not executed.

## Hygiene

- No leaderboard or peer reproduction was inspected or used as evidence.
- Official paper artifacts and generated caches are excluded from git and the
  public Space.
- Secret scanning found no credentials, private records, or local user paths
  in the publishable bundle.
