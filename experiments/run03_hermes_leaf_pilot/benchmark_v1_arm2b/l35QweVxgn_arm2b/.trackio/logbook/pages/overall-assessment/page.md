## Overall assessment

- The paper's **qualitative and order-level content reproduces on CPU**: the `√(K−k)` and `1/n` exponents of Theorem 1, the uniform small-error statement of Theorem 2, the `1/n` decay of Theorem 3, the poly-log-in-T improvement of Theorem 4, and the exact additive decomposition of test-time forgetting all came out as claimed with measured numbers.
- The **one genuine limitation is width**: `m = Ω~(d⁸K⁴)` (≈`3.5e11` at d=16, K=3) is unreachable on CPU, so claims 2 and 3 are capped at *toy* / verified-qualitative with `m = 8d²` substituted. This is a compute limit, not evidence against the paper.
- Claim 2's mutation did **not** break the property even after hardening; that negative result is reported as-is rather than being reframed.
- Three mutations produced predictions that were initially wrong (claims 1, 2, 6); in each case the JSON records the wrong prediction verbatim and the mutation was hardened rather than the verdict softened.
- No claim was **falsified**.

**Reproducibility:** every script is deterministic under `seed = 20260803`; rerun `verify_claim<N>.py` in this directory (`.venv` symlinked to `../20hdQQQrA4/.venv`) to regenerate `results/claim<N>.json`. Per-run stdout is preserved in `log_claim1.txt … log_claim6.txt`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Overall assessment"}\n-->
