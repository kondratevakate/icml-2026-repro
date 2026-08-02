# Claim 5 — safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8e9155155aee", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 5 \u2014 safety-critical control: CAffNet avoids obstacles, HardNet and soft NN fail"}
-->
*Source:* Section 4.3, Table 4, Fig. 6; setup Appendix D.3.
*Type:* simulation (training a differentiable closed-loop rollout). *Script:* `verify_claim5.py` -> `results/claim5.json`.

**Verdict: verified (CPU-scale, reduced initial states).**

Reduced-scale run: 20 initial states, 60 epochs, 3 seeds, CPU.
- CAffNet: 0 collisions, 0 test violations (max: 8.9e-08), mean train violations: 0.14%
- HardNet: 0 collisions, but 1.70 mean test violations (max violation > 0)
- NN (soft): 3 collisions, 3.72 mean test violations (max violation > 0)
- Mutation confirms CAffNet's hard-constraint guarantee vs baselines.

Note: Paper uses 300 initial states + GPU training; this CPU run qualitatively reproduces the safety advantage.

---
