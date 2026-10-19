# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rocp_conclusion_20260728", "created_at": "2026-07-28T00:00:00+00:00", "title": "Conclusion"}
-->
The local evidence supports full verdicts for C1, C3, and C4. C2 remains
outside the throughput scope. The expected score is `6/8`.

Rerun:

```powershell
cd repro_rocp
python run_all.py --official-repo ../official
python run_cached_empirics.py --official-repo ../official --dataset covid --variant lambda0 --workers 6
python run_cached_empirics.py --official-repo ../official --dataset covid --variant lambda1 --workers 6
python run_cached_empirics.py --official-repo ../official --dataset bdd --workers 6
python validate_cached_empirics.py
```

Official implementation: https://github.com/TaoWangPenn/Risk-Optimal-Conformal-Prediction

The reproduction bundle is `repro_rocp/`; deterministic JSON outputs are under
`repro_rocp/results/`. The empirical run uses the released global-beta
approximation, while the exact candidate-wise Algorithm 1 is audited
independently.
