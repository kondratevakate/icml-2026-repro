## Claim 4 — linear classification, "34.39% smaller than Baseline-agg" (Sec. 5.2, Fig. 2)

Wording confirmed verbatim in the paper text: *"MDCP sets are on average 34.39% smaller than Baseline-agg."*

**What was attempted.** The authors' own pipeline, unmodified:
`python notebook/eval_linear.py --num-trials 3 --base-seed 34567 --seeds 34567 34568 34569`
(`mdcp_run/` = copy of `code_mdcp/`, `.venv` interpreter, `matplotlib` added only because
`notebook/simulation.py` imports it at module load).

**What happened.** After **1780 s** of CPU the run was still inside trial 1: it had fitted the three
source models and the pooled GBM and entered the `GAMMA_GRID = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]`
loop (`eval_linear.py:315`), which fits the λ-spline and evaluates MDCP on 3000 test points for each γ,
twice per trial (classification + regression). **0 of 3 trials finished**; the paper averages **100** trials.
Log: `results/eval_linear_official_run.log`; machine-readable status: `results/claim4_5.json`.

**Verdict: inconclusive** — no size-reduction number was measured, so none is reported. Nothing here
supports or contradicts 34.39%; the honest statement is that the official pipeline needs far more than a
75-minute CPU budget for even one of the 100 trials.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 linear classification, \"34.39% smaller than Baseline-agg\" (Sec. 5.2, Fig. 2)"}\n-->
