# SP-Mind publication preflight

Date: 2026-07-29

## Decision

`PUBLISHED`, then revised after independent preflight as a conservative
artifact/implementation reproduction.

Prepared score: **3/10**. The frozen forecast was 4/10; independent preflight
reduced C1 from full to partial implementation verification.

Space:
`https://huggingface.co/spaces/kondratevakate/repro-spmind-artifact-audit`

Rendered logbook:
`https://kondratevakate-repro-spmind-artifact-audit.static.hf.space/`

Hugging Face revision:
`d9284f6a6c1b4be5ab413cd93997bc1d44116a8a`.

The public static Space reports `RUNNING`; both the Space page and rendered
logbook return HTTP 200 with the SP-Mind title. Remote logbook validation
passes.

## Frozen primary artifacts

- paper PDF SHA-256:
  `4CB694513EB6F2B3DC1E47E97E5A9BC53B013FEBE3E963455680802244E10F50`;
- arXiv source archive SHA-256:
  `2B38444FEA77FE5B03F74FC682771785294F63CE20909DFD58ADEAEE9E8CD041`;
- official code commit:
  `d5b889c3649fbdfd1489e5b2f60fc52a0d3ddbc6`;
- SP-Bench manifest SHA-256:
  `286a58a7d5087fe1837c48076f4ddf199cdc89bb4d67f767623a53ce0ff57973`.

## Verification completed

- Paper pages 6 and 7 were rendered and visually inspected.
- Official Python source, benchmark, and evaluator compile successfully.
- A clean Python 3.11 environment installs `spmind==1.0.0` and all declared
  dependencies.
- The CLI help path starts successfully.
- The runtime MCP registry constructs 19 tools.
- Eight independent mutation/unit tests pass.
- A separate high-reasoning preflight reduced C1 to partial verification and
  confirmed that the C5 row-order defect is a qualification, not a full claim
  verdict.
- `prepare_official.py` retrieves and verifies the pinned public code and
  arXiv source required by the deterministic audit.
- Artifact audit and annotation ground-truth audit exit successfully.
- Four official annotation GT files were downloaded (27.6 MB); their schema
  passes and their hashes are recorded in machine-readable evidence.

## Environment blockers recorded

- `ANTHROPIC_API_KEY`: absent;
- exact paper model: retired;
- Docker CLI: present, daemon unavailable;
- NVIDIA GPU: not detected;
- original predictions/traces: absent;
- full official dataset: 16.1 GB and not needed for the verified claims.

These blockers are terminal for C2, C4, and C5 under the frozen stop
conditions. They do not affect deterministic C1/C3 checks.

## Publication hygiene

- No leaderboard or third-party reproduction verdict was used.
- The publishable directory is `repro_spmind/`.
- Paper PDF/source, author repository, datasets, `.venv`, caches, and secrets
  are excluded.
- Paper table values are clearly labeled source concordance, never reproduced
  metrics.
- The annotation row-order finding is labeled an evaluator robustness issue,
  not a claim about the authors’ hidden predictions.
