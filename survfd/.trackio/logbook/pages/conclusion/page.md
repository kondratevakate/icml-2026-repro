# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b42587286d59", "created_at": "2026-07-19T09:09:07+00:00", "title": "Reproduction bundle"}
-->
**Summary.** Four claims tested, all four confirmed, two of them (Claims 1 and 3) with seed-robustness sweeps, and Claim 2's original framing corrected from a hedged "directional" result to a clean confirmation once the right null value was identified. Claim 4 is confirmed at machine precision for ground-truth functions; the fitted-model variant of that claim was not attempted (no scikit-survival in this environment) and is reported as untested, not as passing.

**Reproduction bundle** (the experiment scripts) is attached as the artifact cell below.

**Rerun:**

    python verify_survfd.py              # Claims 1-2, original scenarios
    python audit_seed_fragility.py        # Claims 1-2, 12-seed robustness sweep
    python audit_a3_thm33.py              # Claim 3, Theorem 3.3 (paper's App. A.2 example)
    python audit_a4_localacc.py           # Claim 4, Fig. 2 scenario setup (naive Eq. 10 -- shows the reproducibility caveat)
    python audit_a4_survival.py           # Claim 4, corrected Mobius-based construction

numpy only, CPU, no external data (synthetic 2- and 3-feature scenarios, exact brute-force decomposition).




---
<!-- trackio-cell
{"type": "artifact", "id": "cell_eef9e32bfaf6", "created_at": "2026-07-19T09:11:27+00:00", "title": "Reproduction bundle", "artifact": "repro-functional-decomposition-and-shapley-interactions-for-interpreting-survival-mode/repro-bundle:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `repro-functional-decomposition-and-shapley-interactions-for-interpreting-survival-mode/repro-bundle:v0` · dataset

https://huggingface.co/buckets/kondratevakate/repro-functional-decomposition-and-shapley-interactions-for-interpreting-survival-mode-artifacts#repro-functional-decomposition-and-shapley-interactions-for-interpreting-survival-mode/repro-bundle:v0


