## Claim 4 — piecewise benchmark: CAffNet-TF reduces MSE by 73.33% vs soft NN, zero violations
**Verdict: inconclusive** (direction + the zero-violation half reproduced; the 73.33% magnitude not)
Source: Section 4.1 / Table 2 (paper: NN 0.0045, HardNet 0.0037, CAffNet-FF 0.0020,
CAffNet-TF 0.0012; reduction 73.33%). Script: `verify_claim4.py` -> `results/claim4.json`.
Command actually run: `./.venv/bin/python verify_claim4.py --epochs 20000 --seeds 5`.

Reproduced Appendix D.1 exactly (target f, g1u/g2u/g1l/g2l, A = [1,1,-1,-1]^T,
b = [g1u, g2u, -g1l, -g2l], 50 random train points, 400 linspace test points, Adam lr 1e-4,
full-batch, 5 seeds), **except 20000 epochs instead of the paper's 50000** (CPU budget).

| method | test MSE mean (std) over 5 seeds | viol max (mean over seeds) | seeds with any violation |
|---|---|---|---|
| NN (soft) | **0.001839826621879348** (0.0024093821342899414) | 0.1705633378904027 | 5/5 |
| CAffNet-FF | **0.002884547281501829** (0.0019290806990403236) | **0.0** | 0/5 |
| CAffNet-TF | **0.0009070503751830563** (0.0005683786123839912) | **0.0** | 0/5 |

- Reproduced reduction CAffNet-TF vs NN = **50.69913847335672 %**
  (`reproduced_reduction_TF_vs_NN_pct`) (paper: 73.33 %).
- Reproduced reduction CAffNet-FF vs NN = **-56.783647284944635 %**
  (`reproduced_reduction_FF_vs_NN_pct`) (paper: +55.6 %); FF did **not** beat the soft NN at 20000 epochs.
- Zero-violation sub-claim **verified**: both CAffNet variants have exactly 0.0 max and mean
  violation on all 400 test points in all 5 seeds (`caffnet_zero_violations: true`), while the soft
  NN violates in every seed (`nn_has_violations: true`, `n_seeds_with_any_violation: 5` for NN, `0`
  for both CAffNets). **Mutation** for that sub-claim: removing the CAffine projection layer (= the
  "NN" arm) makes violations appear immediately (NN max violation mean 0.1705633378904027, 5/5 seeds).
- Why inconclusive on the headline number: with 40% of the paper's epochs the absolute MSEs are in
  the paper's ballpark (paper NN 0.0045 vs ours 0.00184; paper TF 0.0012 vs ours 0.00091) and the
  sign of the effect matches (TF lowest MSE, zero violations), but the specific 73.33% figure is not
  recovered and the per-seed std is of the same order as the mean.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 piecewise benchmark: CAffNet-TF reduces MSE by 73.33% vs soft NN, zero violations"}\n-->
