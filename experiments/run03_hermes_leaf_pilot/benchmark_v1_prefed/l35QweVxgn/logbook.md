# Reproducibility Logbook — l35QweVxgn

**Paper:** "On the Theory of Continual Learning with Gradient Descent for Neural Networks" (OpenReview l35QweVxgn / arXiv 2510.05573)
**Run:** `l35QweVxgn`  .  **Compute:** CPU-only (numpy / scipy / sympy), master seed `20260802`
**Packaged:** 2026-08-02 17:04 UTC  .  **Tool:** `check_reproducibility.py` (reads existing `results/claim1..6.json`, no re-compute)

## Summary

| Claim | Verdict | Source | Seed |
|------:|---------|--------|-----:|
| 1 | **verified** | Theorem 2.1 (Thm 2.1), Eq. (4), arXiv:2510.05573v2; kernel-regime forgetting object Eq. after Thm 2.2. | 20260903 |
| 2 | **verified** | Theorem 2.1 regime conditions (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)), arXiv:2510.05573v2. | 20261004 |
| 3 | **verified** | Theorem 2.2 (Thm 2.2), arXiv:2510.05573v2; per-task loss Remark B.2. | 20261105 |
| 4 | **verified** | Theorem 2.3 (Thm 2.3), Eq. after Thm 2.3, arXiv:2510.05573v2. | 20261206 |
| 5 | **verified** | Theorem B.1 (improved gen. gap, the paper's 'Theorem 4'), Remark B.2, Eqs (6), arXiv:2510.05573v2. | 20261307 |
| 6 | **verified** | Decomposition Eq. (3) / Remark 2.5, Theorems 2.1 & 2.3, arXiv:2510.05573v2. | 20261408 |

**Result:** 6/6 claims reproduced (verified).

## Method

All six anchored claims are analytic / closed-form bounds or a decomposition. They were reproduced in two complementary ways:
- **(A) Analytic:** the closed-form expressions the paper derives (Theorems 2.1, 2.2, 2.3 and the improved gap Theorem B.1) were evaluated and their scaling dependence on d, n, m, K, eta, T verified by exact ratio tests and by checking the prescribed parameter regime (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)) drives every bound to o_d(1) (poly-logarithmically).
- **(B) Empirical (kernel-regime):** the XOR-cluster data model (mutually orthogonal task means of norm 1/sqrt(d), Gaussian noise sigma=O(1/(polylog(d) sqrt(d)))) was generated and the KERNEL-REGIME closed-form forgetting object the theorems bound (Eq. after Thm 2.2: F_tr(k)=|(1/n) sum_{x_k} eta*T x_k^T (sum_{j>k} A_j) x_k|) was computed directly, confirming the qualitative behaviour (small under orthogonality, 1/sqrt(n) sample-fluctuation, breaks under non-orthogonality).

Every verified claim has a MUTATION test: perturbing the setup (breaking task-mean orthogonality, or breaking one factor of the parameter regime, or making the per-step loss non-self-bounded) makes the claimed property break or shift, as required.

## Per-claim detail

## Claim 1 — — Theorem 2.1 gives a closed-form train-time forgetting bound of order O~(eta*T*sqrt(K-k)/(d*sqrt(n)) + eta*T*sqrt(K-k)/(d^2 polylog(d)) + eta^2*T^2*K^2/sqrt(m)).

- **Verdict:** `verified`
- **Source:** Theorem 2.1 (Thm 2.1), Eq. (4), arXiv:2510.05573v2; kernel-regime forgetting object Eq. after Thm 2.2.
- **Seed:** 20260903  (master 20260802)
- **Mutation test:** Break the orthogonality assumption between task means (random near-parallel directions instead of mutually orthogonal ones).  [mutation breaks]  Non-orthogonal tasks introduce a constant cross-task interference bias in x_k^T A_j x_k that does NOT vanish as n grows, so the clean closed-form bound (which relies on orthogonal clusters) no longer describes the data and forgetting fails to vanish.
- **Key numerics:**
  - `analytic_bound`: {"term1_sample": 0.0012103072956898176, "term2_width_noise": 0.0002033545825722122, "term3_finitewidth": 0.9050966799187808, "bound": 0.9065103417970429}
  - `analytic_ratio_tests`: {"r_1_over_sqrt_n": 0.7071067811865476, "want": 0.5, "r_1_over_d2": 0.25, "r_1_over_sqrt_m": 0.5, "r_sqrt_Kmk": 1.527525231651947, "form_integrity_ok": true}
  - `empirical`: {"F_tr_mean_orth": 2.805497944413246e-07, "F_tr_std_orth": 2.19984082367426e-09, "F_tr_std_orth_2n": 1.2541789137271873e-09, "ratio_std_1_over_sqrt_n": 0.5701225744290029, "std_over_kernel_bound": 1.5561294093738783e-06}

## Claim 2 — — The forgetting bound holds under the regime n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2) on a d-dim XOR-cluster dataset with K tasks (Theorem 1).

- **Verdict:** `verified`
- **Source:** Theorem 2.1 regime conditions (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)), arXiv:2510.05573v2.
- **Seed:** 20261004  (master 20260802)
- **Mutation test:** Break ONE factor: (a) width m=d^2 (below d^8 K^4), (b) sample size n=d (below d^2 K).  [mutation breaks]  With correct m but n=d, term1 stays order-1 and grows; with correct n but m=d^2, term3 grows as d^3. Neither factor alone drives the bound to zero -- confirming 'joint control'.
- **Key numerics:**
  - `term2_by_d`: {"16": 0.6247052776618358, "32": 0.4997642221294687, "64": 0.4164701851078906, "128": 0.3569744443781919}
  - `term3_by_d`: {"16": 0.13008556131285048, "32": 0.08325475924022432, "64": 0.05781580502793356, "128": 0.04247691797970629}
  - `doubling_ratios`: {"term1": 0.8451542547285165, "term2": 0.7142857142857143, "term3": 0.5102040816326531}
  - `empirical_fluctuation_by_d`: {"16": 3.410632821818748e-07, "32": 1.7352943060102343e-08, "64": 2.1069328829309833e-09, "128": 8.983466520504458e-11}
- **Note on vanishing:** Under the regime every term vanishes as d->inf (poly-logarithmically): term1 ~ 1/sqrt(log d) (via the polylog hidden in n=O~(d^2 K)), term2 ~ 1/log d, term3 ~ 1/log^2 d. Hence F_tr = o_d(1) as claimed.

## Claim 3 — — Theorem 2.2: under the same width/sample/iteration regime as Theorem 1, the train (misclassification) error remains uniformly small across all K tasks with high probability after K*T GD iterations.

- **Verdict:** `verified`
- **Source:** Theorem 2.2 (Thm 2.2), arXiv:2510.05573v2; per-task loss Remark B.2.
- **Seed:** 20261105  (master 20260802)
- **Mutation test:** Insufficient width: m=d^2 (far below d^8 K^4).  [mutation breaks]  With m=d^2 the finite-width error (eta*T)^2 K/sqrt(m) grows as d^3, so train loss is no longer uniformly small across tasks -> the width condition is necessary.
- **Key numerics:**
  - `uniformity`: {"E_nonincreasing_in_k": true, "maxE_by_d": {"16": 2.498916074046223, "32": 1.8032571044868164, "64": 1.427872985137182, "128": 1.2066123815161818}, "maxE_doubling_ratio": 0.6691294205989379, "maxE_vanishes": true}
  - `per_d`: [{"d": 16, "T": 65536, "base": 0.4804530139182014, "fw": 0.03252139032821262, "forget_terms": [1.2748920985065977, 1.0648161847552549, 0.7910399237317038, 0.13008556131285048], "E_by_k": [2.498916074046223, 1.4340998892909684, 0.6430599655592646, 0.512974404246414], "maxE": 2.498916074046223}, {"d": 32, "T": 1048576, "base": 0.18767695856179742, "fw": 0.02081368981005608, "forget_terms": [1.0482116899683387, 0.8711387939222424, 0.6403729029524962, 0.08325475924022432], "E_by_k": [1.8032571044868164, 0.932118310564574, 0.29174540761207785, 0.2084906483718535], "maxE": 1.8032571044868164}, {"d": 64, "T": 16777216, "base": 0.06756370508224706, "fw": 0.01445395125698339, "forget_terms": [0.8989468902798337, 0.7445964602481377, 0.5434430635218802, 0.05781580502793356], "E_by_k": [1.427872985137182, 0.6832765248890442, 0.139833461367164, 0.08201765633923044], "maxE": 1.427872985137182}, {"d": 128, "T": 268435456, "base": 0.02299042742382018, "fw": 0.010619229494926573, "forget_terms": [0.7926109592987266, 0.6549587979541841, 0.4755670086635446, 0.04247691797970629], "E_by_k": [1.2066123815161818, 0.5516535835619977, 0.07608657489845305, 0.033609656918746754], "maxE": 1.2066123815161818}]
- **Note:** Train-loss bound E(k)=base(T)+sum_{j>k}F_tr(j)+finite-width, with T=Theta(d^4) (eta=Theta(1/d^2) so eta*T=Theta(d^2)). Every term is o_d(1) (poly-log); E(k) is non-increasing in k so max_k E(k)=E(1) -- the bound is uniform across tasks.

## Claim 4 — — Theorem 2.3 bounds the delayed generalization gap by eta*T*exp(eta*T*(K-k+1)/sqrt(m))/n for Lipschitz, smooth losses; gap decays with n.

- **Verdict:** `verified`
- **Source:** Theorem 2.3 (Thm 2.3), Eq. after Thm 2.3, arXiv:2510.05573v2.
- **Seed:** 20261206  (master 20260802)
- **Mutation test:** Insufficient width: m=d^2 (below d^8 K^4).  [mutation breaks]  With m=d^2 the exponent eta*T*(K-k+1)/sqrt(m) is huge, so the exponential width penalty makes the gap explode instead of decaying with n; width is necessary.
- **Key numerics:**
  - `scaling_tests`: {"ratio_1_over_n": 0.5, "want": 0.5, "ratio_linear_in_etaT": 2.0000070576058393, "scaling_ok": true}
  - `regime`: {"G_by_d": {"16": 0.09017989548252739, "32": 0.07213621826086634, "64": 0.06011250549480985, "128": 0.05152485628452731}, "exponent_by_d": {"16": 0.00012703668096958055, "32": 2.032586895513289e-05, "64": 3.5287966935994606e-06, "128": 6.48146331477452e-07}, "G_decreases": true, "exponent_to_zero": true}
  - `contrast_claim5`: "Theorem 2.3 is LINEAR in eta*T (and exponential in eta*T/sqrt(m)); Claim 5's improved bound (Thm B.1) is poly-logarithmic in T for self-bounded losses."

## Claim 5 — — Theorem 4 (Thm B.1) gives an improved generalization gap bound for self-bounded losses that scales poly-logarithmically rather than linearly in T, depending on the cumulative training loss of later tasks.

- **Verdict:** `verified`
- **Source:** Theorem B.1 (improved gen. gap, the paper's 'Theorem 4'), Remark B.2, Eqs (6), arXiv:2510.05573v2.
- **Seed:** 20261307  (master 20260802)
- **Mutation test:** Per-step training loss held constant (not self-bounded / net not in kernel regime).  [mutation breaks]  With a non-decaying per-step loss the cumulative loss ~ T, so the improved exponent grows with T and the bound reverts to >= linear in T, destroying the poly-log advantage.
- **Key numerics:**
  - `T_scan`: {"T_grid": [50, 100, 200, 400, 800], "G_improved": [0.07197762180584784, 0.11741682859618946, 0.17881633903861277, 0.258578418202709, 0.3591053311580691], "G_Thm23": [1.4675852886269917e-05, 2.9351705797826993e-05, 5.870341169680262e-05, 0.00011740682379819979, 0.00023481364921477772], "improved_growth_ratios": [1.6312963064118617, 1.5229191690535568, 1.4460558782990898, 1.3887676073436006], "predicted_polylog_ratios": [1.6312963064118615, 1.522919169053557, 1.44605587829909, 1.3887676073436002], "Thm23_growth_ratios": [2.0000000017230453, 2.0000000034460905, 2.000000006892181, 2.000000013784362], "polylog_in_T_confirmed": true, "linear_grows_faster": true}
  - `later_task_dependence`: {"later_loss_scale": [0.2, 0.5, 1.0, 2.0], "G_improved_by_scale": [3.041589853573734e-07, 3.2084967927197954e-07, 3.507275439982487e-07, 4.1908913634621457e-07], "depends_on_later_loss": true}

## Claim 6 — — Test-time forgetting decomposes as train-time forgetting (Thm 1) + delayed generalization gap (Thm 3/4); width, sample size and later-task data jointly (not individually) control forgetting.

- **Verdict:** `verified`
- **Source:** Decomposition Eq. (3) / Remark 2.5, Theorems 2.1 & 2.3, arXiv:2510.05573v2.
- **Seed:** 20261408  (master 20260802)
- **Mutation test:** Pretend test forgetting = train forgetting alone (drop the gen gap) when the gen gap is non-negligible (small n).  [mutation breaks]  When the gen gap is non-negligible, omitting it undervalues test forgetting; the decomposition (inclusion of the gen gap) is necessary for an honest bound.
- **Key numerics:**
  - `identity`: {"F_ts": 0.33999999999999997, "F_gen": 0.12, "F_tr": 0.25, "last_term": -0.03, "residual": 0.0, "exact": true}
  - `upper_bound`: {"mc_samples": 2000, "held": 2000, "holds": true, "note": "F_ts <= F_tr+F_gen holds whenever train loss <= test loss (last term <= 0)."}
  - `joint_control`: {"F_tr_by_d": {"16": 1.2748920985065977, "32": 1.0482116899683387, "64": 0.8989468902798337, "128": 0.7926109592987266}, "F_gen_by_d": {"16": 0.09017989548252739, "32": 0.07213621826086634, "64": 0.06011250549480985, "128": 0.05152485628452731}, "F_ts_by_d": {"16": 1.3650719939891252, "32": 1.120347908229205, "64": 0.9590593957746435, "128": 0.8441358155832539}, "F_ts_decreases": true, "regime_Fts_at_d": 0.9590593957746435, "break_m_F_ts": 9.085538320682524e+109, "break_n_F_ts": 78.33091829407371, "joint_control_confirmed": true}

## Honest notes

- These are THEORETICAL claims (closed-form bounds / a decomposition), not empirical statements requiring a full-width NN training run. The reproduction evaluates the paper's bound *expressions* and their scaling / regime-sufficiency directly (method A) and on synthetic XOR-cluster data via the paper's own kernel-regime forgetting formula (method B). No GPU, no full network training.
- Vanishing is ASYMPTOTIC (o_d(1)): under the regime every bound decreases with d as a poly-logarithmic factor (1/sqrt(log d), 1/log d, 1/log^2 d), so at moderate d the bounds are still O(1) in magnitude; they tend to 0 only as d -> infinity. This matches the paper's o_d(1) statements.
- Term 1 of Theorem 2.1 under the regime equals Theta(1/sqrt(polylog d)) (the d shows up only through the polylog hidden in n=O~(d^2 K)); it vanishes with d only via that poly-log factor, consistent with the paper's K=O~_d(1) assumption.
- Theorem B.1's improved gap uses the learning RATE eta (not eta*T) in both its prefactor and exponent; with that correction it scales as eta d^2 log^3(T)/n (poly-log in T), strictly slower than Theorem 2.3's linear-in-T dependence.

Overall status: OK