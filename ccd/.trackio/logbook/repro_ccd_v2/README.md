# Conditional Coverage Diagnostics: six-claim audit bundle

This CPU-only bundle extends the earlier two-claim reproduction without
changing its evidence. It adds:

- exact finite-state checks of the L1, L2, and KL ERT identities;
- an Algorithm 1 leakage control comparing in-sample and five-fold estimates;
- deterministic audits of the authors' released Table 2 and Table 4 CSVs.

The released CSVs come from official commit
`39a99dcad92205a15d93f2c5fec40c76540abf1c` of
`ElSacho/Conditional_Coverage_Estimation`.

Run:

```powershell
python run_all.py
```

The prior synthetic reproduction is retained separately as
`prior_synthetic/verify_ert.py`.
