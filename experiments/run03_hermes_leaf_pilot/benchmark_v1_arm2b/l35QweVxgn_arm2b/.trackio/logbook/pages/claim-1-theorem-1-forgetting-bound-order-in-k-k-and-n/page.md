## Claim 1 — Theorem 1 forgetting bound (order in (K−k) and n)

**Source recorded:** `"Theorem 2.1 / Eq.4 (Sec 2.2); operationalised via Eq.17 (Sec 2.2, App. C)"`
**Route:** `"analytic functional + Monte-Carlo exponent recovery"`, `d = 40`, `ηT = 1600.0`, `reps_per_cell = 40`.

**Verdict: `verified`** — scope recorded in the JSON as:
> `"order/exponents of the leading (infinite-width) term of Eq.4 in (K-k) and n, recovered from the paper's own Eq.17 functional; the η²T²K²/√m finite-width term is covered in claim 6"`

**Evidence (SNR reading, μ_norm = 1.0):**
- (K−k) sweep `[1,2,4,8,16]` → forgetting `[0.40728, 0.52936, 0.79602, 1.09375, 1.54280]`, `loglog_slope = 0.4889892208910311` against `predicted = 0.5`.
- n sweep `[50,100,200,400,800,1600]` → forgetting `[2.51120, 1.73626, 1.05388, 0.88446, 0.57669, 0.47210]`, `loglog_slope = -0.48797850289581035` against `predicted = -0.5`.
- MC standard errors are small relative to the trend (e.g. `0.0536` at the first gap cell).

**Evidence (literal reading, μ_norm = 1/√d = 0.15811388300841897):** same exponents recovered — `loglog_slope = 0.4275397431077953` in (K−k) and `-0.497929518740411` in n. `"exponents_match": true` for both readings.

**Mutation — non-orthogonal task means (`mutation_nonorthogonal_means`):** `"property_breaks": true`, criterion `"magnitude of the bounded quantity inflates >20x (o_d(1) conclusion destroyed)"`; measured `magnitude_ratio_at_max_gap = 221.1565641039893` (forgetting rises to `[66.84, 93.19, 184.60, 185.79, 341.20]`).
Recorded honestly, verbatim from the JSON:
> `"PREDICTION WRONG, recorded verbatim rather than rewritten: the (K-k) exponent did NOT move to ~1 (measured ~0.57 vs baseline ~0.49). What DID break is the theorem's actual conclusion: the magnitude inflates by ~2e2, so F^tr is no longer o_d(1)."`

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 1 forgetting bound (order in (K\u2212k) and n)"}\n-->
