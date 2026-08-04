## Claim 6 — decomposition `F^ts = F^tr + F^gen`, and *joint* control by width + samples

**Source recorded:** `"Eq. 3 (Sec 2.1.3) + Thm 2.1 discussion (Sec 2.2, 'neither factor alone')"`
**Route:** `"direct measurement of all four quantities + 2-D (m,n) sweep"`. Config: `d=12, K=3, T=40, k=1, ηT=72.0, n_test=4000, reps=8`.

**Verdict: `verified`.**

**Evidence — decomposition identity:** `"identity_exact": true`, `max_identity_residual = 1.0842021724855044e-19` over `n_runs = 8` (all 8 `"interpolating": true`). `frac_inequality_holds_unconditional = 1.0` and `frac_inequality_holds_given_hypothesis = 1.0`. Example run: `F_tr = 0.0023705129680764046`, `F_gen = 0.0004302127194704829`, `F_ts = 0.0010750119899391223`.

**Evidence — joint (m,n) grid, `|F^tr|`:**

| m \ n | 16 | 64 | 1024 |
|---|---|---|---|
| 4 | 0.062142408833394086 | 0.07183102291284485 | 0.04577503819435534 |
| 32 | 0.003362674901524528 | 0.0026523864600899454 | 0.00047904372899644547 |
| 2048 | 0.0029690728357697456 | 0.001053588792873012 | 0.0003390712422030626 |

**Mutation / single-factor arms:** `"property_breaks": true`. From `baseline_small_m_small_n = 0.062142408833394086`: `m_only_ratio = 0.04777852824678765`, `n_only_ratio = 0.736615124094719`, `both_ratio = 0.005456358203173683` — increasing n alone barely helps (26% reduction), while both together give a ~180× reduction, which is the "jointly, not individually" statement.

First-attempt failure logged verbatim:
> `"PREDICTION NOT CONFIRMED on the first grid, recorded verbatim: with m in {d^2,16d^2} and n in {40,640} all four cells gave |F_tr| ~ 5e-4..1.5e-3 and the joint arm was not better than the n-only arm (both_ratio 0.61 vs n_only_ratio 0.58). The grid was hardened (small width where the eta^2T^2K^2/sqrt(m) term is active), not the verdict adjusted."`

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 decomposition `F^ts = F^tr + F^gen`, and *joint* control by width + samples"}\n-->
