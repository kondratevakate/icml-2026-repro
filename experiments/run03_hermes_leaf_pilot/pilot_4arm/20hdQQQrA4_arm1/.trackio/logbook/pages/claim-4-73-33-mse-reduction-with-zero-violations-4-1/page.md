## Claim 4 — 73.33% MSE reduction with zero violations (§4.1)

**Source:** §4.1 / Table 2 (NN 0.0045, CAffNet-TF 0.0012 → 73.33%) and contribution 3 in §1.
Setup from App. D.1, reimplemented in `verify_claim4.py` (+ `claim4_worker.py`,
`claim4_aggregate.py`); raw `results/claim4.json` and `results/claim4_<kind>_<seed>.json`.

Faithful to the paper where specified: n_in = n_out = 1; the exact piecewise target and the
four piecewise bounds of App. D.1; A = [1,1,−1,−1]ᵀ, b = [g1u, g2u, −g1l, −g2l]; 50 uniform
training samples on [−2,2]; 400 linearly spaced test points; MSE loss; soft penalty
100·ReLU(Ay−b); Adam lr 1e-4; **50000 epochs**; batch 500 (= full batch); FF 3×200 ReLU;
TF 3 heads × 40 (d_model 120), feed-forward 120; **5 seeds**. CPU instead of V100.

**Measured (mean over 5 seeds, test MSE against the target function):**

| Method | MSE (mean ± std) | viol max | viol mean | viol % |
|---|---|---|---|---|
| NN (soft) | 1.728e−3 ± 1.853e−3 | 3.336e−1 | 8.42e−4 | 6.41% |
| CAffNet-TF | **7.96e−4** ± 5.26e−4 | **9.19e−8** | 0.0 | 0.06%* |
| TF without CAffine layer (mutation) | 2.557e−3 ± 1.834e−3 | 2.546e−1 | 6.06e−4 | 6.35% |

\* the 9.19e−8 residual is float32 rounding inside the projection, ~6 orders of magnitude
below the baselines; two of five seeds report it on ≤0.25% of test points.

- **Measured MSE reduction NN → CAffNet-TF: 53.94%** (paper: 73.33%).
- Per-seed reduction: **+36.7%, −117.7%, +48.1%, +76.8%, +70.1%** — CAffNet-TF wins on
  4 / 5 seeds; the spread across seeds is larger than the gap between the paper's number and
  ours (the NN baseline's std exceeds its own mean).
- Zero-violation half of the claim: reproduced (3.3e−1 → 9.2e−8, i.e. effectively exact
  feasibility, whereas the soft baseline violates on 6.41% of test points).

**Mutation test.** Removing only the CAffine layer from the *same* transformer (soft penalty
only) restores violations (max 2.546e−1, 6.35% of points) and worsens MSE to 2.557e−3. The
zero-violation property is therefore attributable to the CAffine layer, not to the architecture.

**Verdict: inconclusive** for the anchored 73.33% figure — the direction and the zero-violation
part reproduce, but the magnitude (53.94%) does not match, and with 5 seeds the estimate is
too noisy to call the paper's number either confirmed or refuted. Not marked `falsified`
because this is an independent reimplementation (the authors' code, initialisation and any
unstated training details were not used) and the seed variance is of the same order as the
effect. Sub-claim "zero constraint violations for CAffNet-TF": **verified**.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 73.33% MSE reduction with zero violations (\u00a74.1)"}\n-->
