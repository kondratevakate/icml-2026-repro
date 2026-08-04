## Evidence boundary — what this evidence does NOT cover
- No claim is tested with the paper's original code (not available in the bundle); all
  implementations are re-derivations from the paper text in `notes_paper.md`.
- Claim 1: the bound is validated on random *linear* polyhedra with the exact Eq (12)
  algorithm, not on function classes; the universal-approximation limit (existence of
  f_theta for every eps) is assumed, not tested. No test of the continuity of x -> P*(x).
- Claim 2: the "joint optimization" advantage is shown with a black-box optimiser over w on
  static instances, not with an end-to-end trained w_phi network on the paper's benchmarks.
- Claim 3: instances up to n_out = 5, m = 12; no large-m scaling test, no CAffNet-Lite.
- Claim 4: 20000 epochs (paper 50000), 5 seeds; HardNet arm omitted; timing columns of
  Table 2 (T_train/T_test) not compared; no GPU, so wall-clock numbers are not comparable.
- Claim 5: 8 training epochs and 12 training initial states (paper 300 states), 3 seeds,
  49 evaluation states; PID gains for u_nom are not specified in the paper and were chosen
  by us; HardNet arm and Table 4 numbers are not reproduced; the paper's figures
  (trajectories) are not reproduced.
- Experiment 4.2 (learning optimization solvers, Table 3) was not in the anchored claim list
  and was not attempted.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Evidence boundary \u2014 what this evidence does NOT cover"}\n-->
