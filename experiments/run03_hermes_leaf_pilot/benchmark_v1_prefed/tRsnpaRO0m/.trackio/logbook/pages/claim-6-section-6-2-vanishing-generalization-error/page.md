## Claim 6 — — Section 6.2, vanishing generalization error
Fixed-capacity random-feature MPNN (width 64, depth 3) + ridge head on bofop graphs,
5 replicates per point, m ∈ {50,…,800}, test set 400.
*Process note:* the first run produced gaps of ~1e-7 with a meaningless slope — the head was
fitting the noiseless target essentially exactly, so the gap was at the float64 noise floor and
the rate question was vacuous. Adding label noise σ=0.05 makes the gap a real quantity. With
that, gap decays 2.75e-3 → 2.22e-4, log-log slope **−0.96**, comfortably better than the −1/2
the equicontinuity/covering-number argument guarantees. The claim as stated (error vanishes as
sample size grows) reproduces.
**Caveat, stated plainly:** the heavy-tailed-degree mutation raises the gap level 2–11× at every
m but its slope is *steeper* (−1.54), so it does **not** degrade the asymptotic rate here. The
experiment therefore supports the vanishing-error claim and shows the fiber bound improves the
constant, but it does **not** isolate compactness as necessary for the rate. That part of the
Sec. 6.2 mechanism remains untested by this reproduction.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 \u2014 Section 6.2, vanishing generalization error"}\n-->
