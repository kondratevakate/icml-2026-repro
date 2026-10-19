# MA-RAG independent artifact audit

This package checks the seven released medical-QA JSON files and compares the
released multi-round control flow with the primary paper specification.

Run from the candidate directory:

```powershell
python repro_ma_rag/audit_artifacts.py
python -m unittest repro_ma_rag.test_audit_artifacts
```

The audit is standard-library only. It does not call a language-model API,
download MedCorp, or infer experimental accuracy from the paper tables.
