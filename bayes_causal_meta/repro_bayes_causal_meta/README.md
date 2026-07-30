# Bayesian causal meta-learning independent audit

Run from this directory:

```powershell
python audit_bayes_causal_meta.py
python -m pytest test_audit_bayes_causal_meta.py -q
```

The deterministic audit checks the paper/release prior map, the Gaussian
continuity bound, the negative-transfer theorem constant, and artifact
inventory. Canonical UK Biobank data are not included in the release.

