## Claim 4 — Model-based data augmentation improves horizon dependence to H·[√(log|Π|/n)+eps_q] without RTVC (Theorem 7, §4.2)

*Source:* arXiv:2603.20538, Section 4.2, Theorem 7.
*Type:* theory (bound-formula evaluation). *Script:* `verify_claim4.py` → `results/claim4.json`. *Seed:* 20260324.

**Verdict: verified.**

Bounds evaluated as functions of H (`n=500, log|Pi|=log256, log|M|=log64, eps_q=0.01`):
- model-augmented bound horizon slope = **1.0** (linear `H*eps_q`)
- naive BC *with* RTVC horizon slope = **1.0** (linear)
- naive BC *without* RTVC horizon slope = **1.996** (quadratic `H²`)
- quantization term at H=100: model-augmented = **14.93**, naive-noRTVC = **10011.11**
- improvement ratio at H=100 = **670.48**

The model-augmented formula contains no RTVC modulus κ; it holds under P-EIISS alone, whereas the
naive bound needs RTVC to avoid `H²`. Confirms Theorem 7 removes the smoothness requirement.

**Mutation test (revert model-augmented → naive log-loss BC, RTVC still unavailable):** horizon
dependence jumps from slope **1.00** (linear) to **1.996** (quadratic `H²`) — the claimed improvement
disappears. `passed = true` (|slope_ma−1|<0.05 and |slope_nn−2|<0.05).

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 Model-based data augmentation improves horizon dependence to H\u00b7[\u221a(log|\u03a0|/n)+eps_q] without RTVC (Theorem 7, \u00a74.2)"}\n-->
