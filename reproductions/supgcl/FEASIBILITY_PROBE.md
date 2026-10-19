# SupGCL feasibility probe

Paper: *Supervised Graph Contrastive Learning for Gene Regulatory Networks*
(`0Y7itV1kb7`, arXiv `2505.17786`).

## Decision

`PROCEED` for a CPU theorem/source audit. `DEFER` the empirical tables until
the official graph archive and suitable GPU compute are available.

## Available artifacts

- official code: `https://github.com/shobioinfo/SupGCL`;
- pinned commit: `a384eeaeffaabfd140335e19a0633ab400ba3bcb`;
- official processed data: `https://zenodo.org/records/15496012`;
- arXiv source archive SHA-256:
  `20d8c3ef3ce61e816e88817d2cccabddd7b0fc8e8d3b0c3fa2ba1436b5e55ec8`;
- complete code for pretraining, five original baselines, additional baselines,
  and five downstream scripts.

## Feasible locally

- reconstruct Theorem 1 as a finite KL chain-rule identity;
- reconstruct Corollary 1 using independent softmax distributions;
- audit the pinned code path from LINCS knockdown metadata to the teacher
  sampler;
- publish deterministic synthetic evidence without medical records.

## Deferred

The official README specifies a roughly 4.3 GB data archive and 3000
pretraining epochs. The paper reports a single H100 GPU, 10-fold
cross-validation for downstream tasks, and 10 random seeds for an imbalanced
classification task. Tables 2-3 and the embedding analysis are therefore not
local high-throughput claims.

No leaderboard or peer reproduction was inspected.
