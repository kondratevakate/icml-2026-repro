## Claim 6 — — Test-time forgetting decomposes as train-time forgetting (Thm 1) + delayed generalization gap (Thm 3/4); width, sample size and later-task data jointly (not individually) control forgetting.

- **Verdict:** `verified`
- **Source:** Decomposition Eq. (3) / Remark 2.5, Theorems 2.1 & 2.3, arXiv:2510.05573v2.
- **Seed:** 20261408  (master 20260802)
- **Mutation test:** Pretend test forgetting = train forgetting alone (drop the gen gap) when the gen gap is non-negligible (small n).  [mutation breaks]  When the gen gap is non-negligible, omitting it undervalues test forgetting; the decomposition (inclusion of the gen gap) is necessary for an honest bound.
- **Key numerics:**
  - `identity`: {"F_ts": 0.33999999999999997, "F_gen": 0.12, "F_tr": 0.25, "last_term": -0.03, "residual": 0.0, "exact": true}
  - `upper_bound`: {"mc_samples": 2000, "held": 2000, "holds": true, "note": "F_ts <= F_tr+F_gen holds whenever train loss <= test loss (last term <= 0)."}
  - `joint_control`: {"F_tr_by_d": {"16": 1.2748920985065977, "32": 1.0482116899683387, "64": 0.8989468902798337, "128": 0.7926109592987266}, "F_gen_by_d": {"16": 0.09017989548252739, "32": 0.07213621826086634, "64": 0.06011250549480985, "128": 0.05152485628452731}, "F_ts_by_d": {"16": 1.3650719939891252, "32": 1.120347908229205, "64": 0.9590593957746435, "128": 0.8441358155832539}, "F_ts_decreases": true, "regime_Fts_at_d": 0.9590593957746435, "break_m_F_ts": 9.085538320682524e+109, "break_n_F_ts": 78.33091829407371, "joint_control_confirmed": true}

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 \u2014 Test-time forgetting decomposes as train-time forgetting (Thm 1) + delayed generalization gap (Thm 3/4); width, sample size and later-task data jointly (not individually) control forgetting."}\n-->
