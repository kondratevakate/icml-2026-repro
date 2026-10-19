# Claim 4: Claim A1: theorem 4.1 establishes exponential decay of finite-sample bias of KM-ARL with N


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f49e4a79adeb", "created_at": "2026-07-20T08:30:51+00:00", "title": "Claim 4: Claim A1: theorem 4.1 establishes exponential decay of finite-sample bias of KM-ARL with N"}
-->
**Anchored claim (verbatim).** "Theorem 4.1 establishes that the finite-sample bias of the KM-ARL estimator decays exponentially with dataset size N."

**Verdict -- VERIFIED under Theorem 4.1's stated assumptions.**

**Independent derivation audit.** I checked the proof from the estimator definition rather than fitting a decay curve. Integration by parts gives the paper's Eqs. (31)-(36), and therefore the finite-sample bias identity `B_FS = I1 - a I2` in Eq. (37). Applying the cited Stute KM inequality once with `phi(t)=1[t<=a]` and once with `phi(t)=t*1[t<=a]` gives the stated signs and bounds for `I2` and `I1`; combining their worst cases gives Eq. (8). The only `N`-dependent term is `H(t)^(N-1)`, so the bound contracts exponentially wherever the theorem's stated condition `H(t)<1` holds on `[0,a]`.

**Independent numerical audit.** `audit_finite_sample_bounds.py` uses continuous independent exponential event and censoring laws, evaluates both sides of Eq. (8) by 256-node Gauss-Legendre quadrature, and estimates the expected truncated KM mean with **500,000 independent datasets per cell**. For `N in {2,4,8,16,32,64}`, every Monte Carlo interval `bias +/- 5 SE` lies strictly inside the analytic theorem bounds. Measured bias contracts from `+0.574052` to `+0.001644`; the total bound width contracts from `2.432203` to `0.094356` (25.8x), while the integrand retains the theorem's explicit `H(t)^(N-1)` factor.

**Destructive control.** Setting `censor = 0.5 * event` preserves continuous marginals but deliberately removes independent censoring. At `N=64`, plugging those marginals into the theorem gives `[-0.490814, +0.628612]`, while the observed bias is `+6.832624`: the upper bound fails by more than 10x. This shows the positive audit is sensitive to the theorem's mechanism rather than a vacuous wide interval.

**Evidence boundary.** The audit does not re-prove Corollary 1.1 of Stute (1994), which is the external lemma explicitly invoked by the paper. It independently verifies the paper's mapping from that lemma to Theorem 4.1, the bound arithmetic, the exponential factor, and one continuous exponential-distribution family with a mechanism-breaking control.


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_f9d0f4076e22", "created_at": "2026-07-20T08:33:00+00:00", "title": "Dashboard: repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis", "dashboard_project": "repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis"}
-->
**🎯 Trackio dashboard** `repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis`

trackio-local-dashboard://repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis
