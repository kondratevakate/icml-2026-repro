# Claim 6 — Section 5, Figures 5 and 6: 2D locomotion simulation + real robot experiments

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ca1d0cc35d68", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 6 \u2014 Section 5, Figures 5 and 6: 2D locomotion simulation + real robot experiments"}
-->
**Verdict: `inconclusive`** (hardware unavailable; no toy substitute produced).

*Source:* Figures 5 and 6 (and Figure 3). *Script:* `verify_claim6.py` → `results/claim6.json`.

* Humanoid vertical jump and quadruped bounding (Fig. 6) require physical robots; no code, robot
  model, logs or rosbags were released, and the tracking layer is an external DDP implementation
  (Crocoddyl) with unreported cost weights. Figure 3's dynamic-violation curves come from that
  same unreleased pipeline. Not reproducible here, and no simulated stand-in was presented as if
  it were the hardware result.
* The 2D-locomotion-simulation half is *partially* covered by claim 4 (eq. 6 implemented, linear
  convergence observed across Δt, forces box-constrained), but the specific curves of Figure 5
  cannot be matched because T, the target CoM area, the contact schedule and locations r_i^j, the
  initial state and ρ are not reported.

---
