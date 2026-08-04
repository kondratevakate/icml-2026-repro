## Summary of honest limitations

1. **5 of 8 datasets** (HUMAINE, MT-Bench, WD, HiFiC, ConHa) are not obtainable → claims 2, 3, 4 can only be checked partially or against the paper's own table.
2. **The official Crowd-BT implementation** (Google) was not used; ours is a faithful-in-spirit but not identical SGD reimplementation. Any Crowd-BT number here is ours, not the paper's.
3. **The public IHQ release is smaller** than the paper's own dataset table, so exact percentages/correlations on IHQ are unreachable by construction.
4. **The IHQ gt** (CLIC 2024 leaderboard) is unavailable; the paper's fallback gt was used instead.
5. Bootstrap/trial counts were reduced (500 / 200 / 1,000 instead of 10,000) to fit the CPU budget; all reported quantities are stable to the third digit under this reduction except claim-3 top-1 percentages, which carry ≈±2 pp Monte-Carlo error.
6. Two **paper-internal inconsistencies** surfaced and are reported rather than smoothed over: the 3.29 vs 2.576 confidence constant (claim 5) and the "second on the remaining three" statement contradicted by Table 1 (claim 4).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary of honest limitations"}\n-->
