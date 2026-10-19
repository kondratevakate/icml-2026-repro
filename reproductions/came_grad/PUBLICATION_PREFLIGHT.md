# CAME-Grad publication preflight

Date: 2026-07-29

## Decision

`PUBLISHED` as an equation-level and release-completeness audit.

Prepared score: **2/10**, matching the frozen forecast.

Space:
`https://huggingface.co/spaces/kondratevakate/repro-came-grad-equation-audit`

Rendered logbook:
`https://kondratevakate-repro-came-grad-equation-audit.static.hf.space/`

Hugging Face revision:
`9639983d6636db9372264db1253d40a823a55463`.

The public static Space reports `RUNNING`; both URLs return HTTP 200 with the
CAME-Grad title, and remote logbook validation passes.

## Frozen artifacts

- official code commit:
  `79059e39060d13ef6b6cb2ea9ad7a5d8e2519c83`;
- paper PDF SHA-256:
  `03D9097B1F9B5E39DE89703385586123A87DCBFBF574207A89ECE58A01F28295`;
- source archive SHA-256:
  `9F5109704136930A9453E0995AAD73EC6F794F51FF15C55E54A3F031369199D1`;
- main TeX SHA-256:
  `32a52a8b12f4c9b71bb48aa64cf87340ae2336dd3f08321c4e8557c84d26208c`.

## Verification

- Key PDF pages 5–9 were rendered; Algorithm 1 and Tables 1–4 were visually
  inspected.
- Five deterministic unit/mutation tests pass.
- The independent equation audit completes 500 random problems.
- Published 2.3/1.9 percentage-point averages are arithmetically concordant
  with table cells, but labeled non-reproduction.
- All released Python files pass AST/bytecode compilation.
- Official entrypoints and imports fail for recorded missing dependencies and
  artifacts.

## Hygiene

- No leaderboard or third-party verdict contributes evidence.
- The initial file search touched local verdict aggregates; their contents were
  explicitly discarded before claim work.
- Official PDF/source/code and generated caches are excluded from the
  publishable bundle.
- Synthetic gradients are used only for equation invariants and
  counterexamples, never as radiology-performance evidence.
