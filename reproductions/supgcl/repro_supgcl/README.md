# SupGCL reproduction

This CPU-only bundle independently audits six claims from *Supervised Graph
Contrastive Learning for Gene Regulatory Networks* (OpenReview `0Y7itV1kb7`,
arXiv `2505.17786`).

Prepared score: **4/12**.

## Results

- C1 `VERIFIED` (2/2): an independent finite-distribution calculation verifies
  the KL chain-rule decomposition in Theorem 1. Pinned author code also
  confirms that LINCS knockdown graphs and knockdown-gene metadata drive the
  supervised augmentation sampler.
- C2 `VERIFIED` (2/2): independent exhaustive calculations verify that both
  augmentation distributions become uniform as `tau_a` grows, the
  augmentation KL vanishes, and the SupGCL loss converges to the uniformly
  augmentation-averaged node loss.
- C3-C6 `NOT EXECUTED` (0/8): reproducing Tables 2-3 and the embedding analysis
  requires the 4.3 GB graph archive, long GPU pretraining, and repeated
  downstream cross-validation.

## Run

```bash
python prepare_official.py
python audit_claims.py \
  --code-root ../official/code \
  --paper-source ../official/paper \
  --output evidence/claims_audit.json
python -m unittest discover -s tests -v
```

The audit uses deterministic synthetic probability tables, not paper-reported
metrics or cached author outputs. It requires only NumPy and a CPU.

## Scope

Theorem 1 and Corollary 1 are verified as mathematical statements. This does
not verify that the released stochastic estimator is an exact implementation
of the full objective, nor does it verify any TCGA downstream result.

No leaderboard or peer reproduction was inspected or used.

Primary sources: [arXiv](https://arxiv.org/abs/2505.17786),
[OpenReview](https://openreview.net/forum?id=0Y7itV1kb7),
[official code](https://github.com/shobioinfo/SupGCL), and
[official data](https://zenodo.org/records/15496012).
