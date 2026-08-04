## Claim 1 — — verified

**Verdict:** sec. 3.3, eqs. 11–13, app. a

*Source:* Sec. 3.3, Eqs. (11)–(13); derivation App. A. *Script:* `verify_claim1.py` (15 s).

- **100 EM runs** (25 synthetic datasets × 4 random inits, data drawn from Eq. 2): **0 monotonicity violations** of the log-posterior; worst single-step change `+3.6e-3` (i.e. always increasing).
- **Stationarity** at convergence: max |∂ log-posterior| = `5.7e-8` (relative `5.9e-11`); EM fixed-point residual `2.4e-12` (λ), `5.6e-14` (q).
- **Contrast:** our Crowd-BT gradient run decreases its own objective on **16 of 39** epoch transitions (worst step −2.40) — no monotonicity, as the paper argues.
- **Mutation:** replacing the E-step responsibility γ (Eq. 11) with a mis-specified `clip(γ^0.35·1.4)` breaks monotonicity in **10/10** datasets (worst step −17.26). ✔ property is load-bearing.
- *Caveat:* this is an empirical check of an analytic guarantee, not a proof. The guarantee is the standard EM result (Dempster et al. 1977) and the M-step updates are exact maximisers of Q, so this is expected.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 \u2014 verified"}\n-->
