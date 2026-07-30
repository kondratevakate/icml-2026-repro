# Official artifacts

Pinned before execution on 2026-07-29.

## Paper

- arXiv abstract: `https://arxiv.org/abs/2511.19953`
- arXiv PDF: `https://arxiv.org/pdf/2511.19953`
- arXiv source: `https://arxiv.org/e-print/2511.19953`
- PDF SHA-256: `bf31eea0e1312268f3dbaee43f821cc4d90ac5e6bba57d149e6899989339d1b0`
- source archive SHA-256: `d7abb8b085d2ea5183e69a32509d332967af9f45ecff33f1e5cf7fda41e6927f`

## Code

- repository: `https://github.com/Y-Research-SBU/SPROUT`
- pinned commit: `ee23c5fb41c9fe302a28883a08935b674639d7cb`
- tracked files: 40

The audit uses the commit above, not the moving default branch.

## Referenced external artifacts

- the README's `Dataset Donloads` Markdown target is empty;
- the default feature model is `hf-hub:bioptimus/UNI2-h`, whose Hub repository
  does not resolve at freeze time;
- SAM2 code and checkpoints are not vendored; the README instructs replacing a
  directory with Meta's external SAM2 repository;
- no release predictions, metric tables, or checkpoints are present.

These are recorded availability checks, not inferred substitutes. The audit
does not silently replace the default backbone or reconstruct an unspecified
preprocessed dataset.

## Scope

Official artifacts are evidence inputs, not ground truth. Paper claims are
scored only when the released artifacts permit an independent check. Public
leaderboards, third-party verdicts, and prior logbooks are excluded.
