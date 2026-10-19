# Release audits: what blocks verification of claims

**Question.** In the author's own release audits of ICML 2026 papers, what kept claims from being fully
checked: missing code, missing data, or missing intermediate artifacts and weights?

**Protocol.** [`PROTOCOL.md`](PROTOCOL.md) (titled "Protocol 3" in this folder), frozen in commit `cd3e22b`
before any claim was coded.

**Inputs.** The four audit documents (`OFFICIAL_ARTIFACTS.md`, `FROZEN_FORECAST.md`,
`CLAIM_DECOMPOSITION.md`, `INDEPENDENT_CHECKS.md`) in 16 paper folders, now under
[`../../reproductions/`](../../reproductions/INDEX.md): 3dmedagent, bayes_causal_meta, came_grad, caml,
dc_pnpdp, dpsurv, flow_density_ratio, glean, globalhealthatlas, lvcg, ma_rag, medcrp_cl, ophiuchus, spmind,
sprout, xmae.

**Coding.** Coder 1 is the analysis session (`coder1_codes.py` writes its codes as data with consistency
checks); coder 2 is an independent Claude subagent given only the protocol and the folder paths. Codes:
`papers_coder1.csv`, `papers_coder2.csv`, `claims_coder1.csv`, `claims_coder2.csv`; disagreements resolved
after agreement was computed (`resolve.py` → `papers_resolved.csv`, `claims_resolved.csv`).

**Run.**

```bash
cd verdict_audit/healthcare
python analyze.py          # agreement, then Q1-Q4 for each coder and for the resolved codes
```

**Outputs.** [`results.json`](results.json): agreement and Q1–Q4 per coder and resolved.

**Prespecified label.** Q1 "code release did not make claims checkable": met for both coders and for the
resolved codes. Abstract 2 uses `resolved.Q1` (62% of claims that could not be fully checked lacked
intermediate artifacts or weights).

**Limits.** The audits are the author's own, on papers chosen by interest and feasibility; coders are
language models.
