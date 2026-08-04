## Claim 1 — Theorem 4.1 (double-chain, i.i.d.)
`verify_claim1.py` → `results/claim1.json`. n=8, d=3, η₁=0.1100.
- **Sample-independent fixed point:** 8 runs with different trajectories *and* different random initializations (‖θ₀‖≈3·√d), α=1e-3, T=1e5, tail-averaged over the last 50k iterates: max coordinate spread across runs **0.0284**, max distance to the analytic θ* **0.0497** (θ*=[0.5124, 0.8737, 0.3496]). θ* from the expected update equals the Eq.-(14) solution to 2.2e-16.
- **Sample complexity:** η swept 0.1100→0.0185 by conditioning the third feature column; α = Θ(ε·η); hitting times 12 401 → 259 519. Log-log fit **exponent −1.64** (R²=0.991) vs the theoretical upper bound −2. The measured cost grows *no faster* than η⁻², i.e. consistent with Õ(ε⁻¹η⁻²).
- **Mutation:** reuse the same sample for the second chain (destroying the independence that fixes the double-sampling bias) → the iterates settle **1.071** away from θ* (vs 0.05 for the correct algorithm). The claim's mechanism is load-bearing.
- Verdict: **verified**.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 4.1 (double-chain, i.i.d.)"}\n-->
