# DC-PnPDP independent audit

Run from this directory:

```powershell
python audit_dc_pnpdp.py
python -m pytest test_audit_dc_pnpdp.py -q
```

The audit uses the pinned official repository under `../official/repo`. It does
not download clinical images or execute the canonical CUDA reconstruction.

