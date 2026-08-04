## Claim 1 — BC with quantized actions + log-loss matches lower bounds up to the quantization term (Theorem 2, §3.2)

*Source:* arXiv:2603.20538, Section 3.2; lower bounds Section 5 (Theorems 8–9).
*Type:* theory + direct experiment. *Script:* `verify_claim1.py` → `results/claim1.json`. *Seed:* 20260320.

**Verdict: verified.**

Two complementary checks:
**(A) Analytic rate matching.** The statistical part of the log-loss BC upper bounds has the
*same* `n`-dependence as the minimax lower bounds (Thm 8 deterministic = `1/n`; Thm 9
stochastic = `sqrt(1/n)`). The *only* extra term is `H*eps_q`, which the lower bounds themselves
prove unavoidable. Stored as the mean upper/lower ratio over a `n`-grid:
- deterministic upper/lower ratio mean = **102.49** (≈ constant in `n`)
- stochastic upper/lower ratio mean = **2.922** (≈ constant in `n`)
- mutation `eps_q→0`: ratio stays **102.49** (still constant) → the two rates match exactly;
  quantization error is the sole residual gap.

**(B) Direct log-loss BC experiment.** MLE (log-loss) over a finite policy class of size 256,
`n_grid` of 12 points (10 → 3162), 40 repeats, excess log-loss vs `n`:
- fitted log–log slope = **−0.9198** (expected −1.0)
- state-blind baseline slope = **0.0032** (≈ 0 → does not match the lower bound)
- excess risk monotone decreasing: 0.6774 → … → 0.0038 (confirms the `1/n` statistical rate is actually achieved).

**Mutation test (perturb quantization error and estimator):** `eps_q→0` keeps the upper/lower
statistical ratio constant (claim's "up to quantization error" qualifier is the only slack);
replacing log-loss MLE by a state-blind estimator stops the `1/n` decay (slope ~0.00), so the
"matching lower bound" property breaks → confirms the log-loss objective is necessary. `passed = true`.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 BC with quantized actions + log-loss matches lower bounds up to the quantization term (Theorem 2, \u00a73.2)"}\n-->
