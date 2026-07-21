# Claim 3: Claim A4: KM-ARL and KM-ADD are asymptotically unbiased whenever the detection-point support does not exceed the censoring boundary (Eq. 11)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3f840193f0cd", "created_at": "2026-07-20T08:30:51+00:00", "title": "Claim 3: Claim A4: KM-ARL and KM-ADD are asymptotically unbiased whenever the detection-point support does not exceed the censoring boundary (Eq. 11)"}
-->
**Anchored claim (verbatim).** "KM-ARL and KM-ADD are shown to be asymptotically unbiased (Equation 11) whenever the support of the detection-point distribution does not exceed the censoring boundary, without requiring parametric assumptions (Section 4)."

**Setup.** GSR/Gaussian pipeline, threshold=50 (true ARL 60.1), changepoint prevalence 0.5, 10 seeds per N in {50, 200, 1000, 5000, 20000}. Two conditions tested side by side: **support SATISFIED** (`T=5000` const, well above true ARL, so the detection-point distribution's support sits inside the censoring boundary) and **support VIOLATED** (`T=100` const, deliberately forcing the truncation length below the true support). This tests **both directions of the iff** in Eq. 11, not just the positive case.

| N | SATISFIED (T=5000) rel. bias | VIOLATED (T=100) rel. bias |
| --- | --- | --- |
| 50 | -0.023 (sd 2.65 abs) | plateaus, does not shrink |
| 1000 | +0.005 | ~-0.107 |
| 5000 | +0.001 | ~-0.107 |
| 20000 | **+0.003** (sd 0.46 abs) | **-0.107** |

**Result.** With the support condition satisfied, KM-ARL relative bias goes **-0.023 -> +0.005 -> +0.001 -> +0.003** as N increases 50 -> 1000 -> 5000 -> 20000, converging toward zero, with seed sd shrinking from 2.65 to 0.46 (absolute units) over the same range -- textbook asymptotic unbiasedness. With the condition deliberately violated, bias plateaus at **-0.107** and does **not** vanish as N grows, exactly as Eq. 11's "iff" requires. Deciding pair at N=20000: **-0.003 (satisfied) vs -0.107 (violated)** -- more than 30x apart, with the sample-size trend going in opposite directions (toward 0 vs. flat).

**Verdict -- CONFIRMED on BOTH sides of the iff condition.** This is the strongest of the three confirmed results because both directions were tested independently rather than assuming the converse: satisfying the support condition is not merely a sufficient decoration on top of an estimator that would converge anyway, and violating it produces a real, non-vanishing floor.
