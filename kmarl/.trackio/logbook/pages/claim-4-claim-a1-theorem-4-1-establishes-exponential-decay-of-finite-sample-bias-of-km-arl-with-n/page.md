# Claim 4: Claim A1: theorem 4.1 establishes exponential decay of finite-sample bias of KM-ARL with N


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f49e4a79adeb", "created_at": "2026-07-20T08:30:51+00:00", "title": "Claim 4: Claim A1: theorem 4.1 establishes exponential decay of finite-sample bias of KM-ARL with N"}
-->
**Anchored claim (verbatim).** "Theorem 4.1 establishes that the finite-sample bias of the KM-ARL estimator decays exponentially with dataset size N."

**Verdict -- NOT ATTEMPTED numerically / INCONCLUSIVE.** Stated honestly rather than forced to a verdict it does not support: the paper itself notes that empirically verifying this exponential decay is hard because at small N, estimation variance overshadows the bias signal that would need to be fit to an exponential curve. This is directly confirmed by the A4 sweep above -- at N=50, seed standard deviation of the KM-ARL bias is **2.65**, larger than the mean bias itself (**-1.35** midway between the satisfied/violated conditions' small-N values), so any attempt to fit a decay rate from these points would be fitting noise, not signal.

**What the reproduction does show:** bias decay toward zero is observed (see Claim A4: -0.023 -> +0.005 -> +0.001 -> +0.003 as N grows from 50 to 20000 under the satisfied condition), consistent with the theorem's qualitative direction. But the **exponential rate** claimed by Theorem 4.1 -- as opposed to mere convergence to zero -- is not established by this reproduction. Doing so honestly would require either (a) enough seeds at each N for the bias estimate's own standard error to shrink well below the bias itself even at small N, or (b) a much larger sweep of N values fit against an exponential-vs-polynomial null, neither of which was run here.

**This claim is explicitly NOT claimed as confirmed.**


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_f9d0f4076e22", "created_at": "2026-07-20T08:33:00+00:00", "title": "Dashboard: repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis", "dashboard_project": "repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis"}
-->
**🎯 Trackio dashboard** `repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis`

trackio-local-dashboard://repro-accurate-evaluation-of-quickest-changepoint-detectors-via-non-parametric-survival-analysis
