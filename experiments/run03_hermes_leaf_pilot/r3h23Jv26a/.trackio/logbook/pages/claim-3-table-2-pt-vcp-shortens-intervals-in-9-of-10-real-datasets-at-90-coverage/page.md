# Claim 3 — Table 2: PT-VCP shortens intervals in 9 of 10 real datasets at 90% coverage

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5a7044597a1d", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 3 \u2014 Table 2: PT-VCP shortens intervals in 9 of 10 real datasets at 90% coverage"}
-->
**Source:** Table 2 (§3.4); setting App. D.2.1 / D.2.3 (3-layer 64-64 MLP, Adam 5e-4, bs 64, wd 1e-6, dropout 0.1,
α=0.1, p=0.95, 5 seeds, per-dataset output bias). **Script:** `verify_claim3.py` → `results/claim3.json`
(runtime 1993 s CPU). **Verdict: `inconclusive` for the "9 of 10" statement; partial reproduction on 6 of 10 datasets.**

Datasets attempted (all UCI, downloaded into `./data/`): bike, bio (CASP), concrete, blog-data, facebook-1, facebook-2.
**Not attempted:** meps-19/20/21 (AHRQ MEPS public-use files require registration/data-use agreement plus the CQR
repo's multi-step preprocessing) and star (Harvard Dataverse, gated download). Because 4 of the 10 datasets — and in
particular all three MEPS datasets the anchored claim names — are missing, the literal "9 of 10" count cannot be
confirmed or refuted; it is **not** substituted with a toy.

Result on the 6 attempted datasets (5 seeds, mean ± SEM; paper values in brackets):
| dataset | bias | VCP cov | VCP len | PT cov | PT len | PT shorter? | paper |
|---|---|---|---|---|---|---|---|
| bike | 10 | 0.896 | 20.98±0.03 | 0.896 | **20.24±0.03** | yes | 20.46 → 19.59 (yes) |
| bio | 10 | 0.908 | 21.46±0.02 | 0.903 | **20.66±0.06** | yes | 21.13 → 20.44 (yes) |
| concrete | 5 | 0.888 | 10.55±0.06 | 0.896 | **10.21±0.12** | yes | 10.32 → 9.87 (yes) |
| blog-data | 20 | 0.903 | 43.12±0.46 | 0.898 | **42.72±0.63** | yes | 41.67 → 41.13 (yes) |
| facebook-1 | 10 | 0.895 | 21.98±0.38 | 0.899 | 22.67±0.47 | **no** | 20.81 → 20.80 (tie/yes by 0.01) |
| facebook-2 | 10 | 0.899 | 22.54±0.43 | 0.899 | 23.56±0.68 | **no** | 20.97 → 21.01 (**no** — the paper's 1/10 failure) |

**4 of 6 shorter**, and marginal coverage is within 0.02 of 0.90 on every dataset and both methods
(`all_coverage_within_0.02_of_nominal: true`), so the coverage half of the claim reproduces cleanly.
The two failures land on exactly the datasets the paper itself reports as marginal/negative (facebook-1 is a
0.01 tie in Table 2, facebook-2 is the paper's single negative case), so the qualitative picture agrees; my
facebook-1 goes negative rather than tied, which would make the count 8/10 rather than 9/10 if the pattern held.
Table 3's IS result also reproduces here: IS(PT-VCP) ranges 5.46–95.83 across the six datasets while IS(VCP) = 0.

Deviations from the paper's protocol (CPU budget): ≤8000 rows subsampled per dataset, 40/20/40 train/cal/test split,
≤60 epochs with 10%-validation early stopping (patience 8) instead of cross-validated ≤1000 epochs, features
z-scored on the training fold and the response divided by mean|y_train| (CQR-style). No mutation test is reported
for this claim because it is not marked `verified`.

---
