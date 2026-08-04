## Evidence boundary — what this evidence does NOT cover

- **Nothing here is a proof.** Claims 1–3 are numerical over a finite sampled instance space
  (random Gaussian A with prescribed rank, ‖a_i‖ = 1, n_out ≤ 5, m ≤ 12). No symbolic or
  machine-checked verification of Theorem 3.5, Lemma 3.3 or Theorem 3.4 was performed, and
  no adversarial search for counterexamples (only random + exhaustive-over-grid sampling).
- **Ill-conditioning untested.** All pseudo-inverses were on well-conditioned normalised
  matrices; behaviour under near-degenerate A_γ (σ_min → 0), where `pinv` cutoffs matter, is
  not characterised. Feasibility was judged with tolerance 1e−9.
- **Assumption 3.2 always held by construction** (S(x) non-empty and A, b continuous). The
  infeasible-constraint case is out of scope, as it is for the paper.
- **Claim 4** is an independent reimplementation on CPU, not the authors' code: initialisation
  scheme, exact transformer block layout (I used `nn.TransformerEncoderLayer` with d_model 120,
  3 heads, ff 120, no dropout), sampling of the 50 training points, and any unstated tricks may
  differ. Only 5 seeds; float32. CAffNet-FF, HardNet, the training/inference timings of
  Table 2, and Fig. 5's convergence-rate claim were **not** measured.
- **Claim 4's null-space component is inert** in this benchmark: n_out = 1 ⇒ every A_γ is
  full column rank ⇒ I − A_γ^†A_γ = 0. So §4.1 provides no evidence about w_φ; that evidence
  comes only from claim 2's synthetic tasks.
- **Claim 5**: no policy was trained; no cost value comparable to Table 4 was produced; the
  safety constraint, reference and dt are mine, not the paper's. Obstacle *avoidance* as a
  non-convex specification is outside the class of constraints CAffNet is defined for — the
  paper's own reduction of §4.3 to affine constraints is undocumented.
- **§4.2 (learning optimization solvers, Table 3), the IPOPT comparison, CAffNet-Lite, all
  runtime/scalability claims and Fig. 3/4/5** were not anchored claims and were not examined.
- No GPU was used, so nothing here speaks to the reported training/inference times.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Evidence boundary \u2014 what this evidence does NOT cover"}\n-->
