# Reproduction capsule

This capsule builds the static Trackio logbook for the MedCRP-CL release audit.
It consumes `../evidence/audit_results.json`, produced by
`../audit_medcrp_cl.py`.

```powershell
cd medcrp_cl
python audit_medcrp_cl.py
python -m unittest discover -s tests -v
python repro_medcrp_cl/build_logbook.py
python ..\validate_icml_logbook.py
```

The official paper, source, repository clone, and checkpoints live under
`../official/` and are intentionally excluded from version control.
