# GlobalHealthAtlas independent audit

This package audits the public artifacts for arXiv:2602.00491 at frozen
revisions. It does not call commercial models, use third-party verdicts, or
claim to regenerate the unreleased 280,210-instance corpus.

Run:

```powershell
python audit_artifacts.py --output evidence/artifact_audit.json
python -m pytest -q
```

The audit independently:

- reconstructs benchmark aggregates from the detailed per-model CSVs;
- recomputes incremental-SFT, transfer, robustness, and leakage table values;
- checks both released LoRA adapter payloads against their Hub LFS SHA-256;
- parses safetensors headers without loading model weights;
- exercises the lightweight prompt and atomic JSON utilities; and
- records which paper claims cannot be checked because row-level artifacts
  were not released.

