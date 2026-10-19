# SupGCL claim decomposition

## Prepared claims

| ID | Claim | Method | Verdict | Points |
| --- | --- | --- | --- | ---: |
| C1 | Theorem 1 KL decomposition and LINCS supervision | independent finite probability tables plus pinned source anchors | Verified | 2 |
| C2 | Corollary 1 recovers uniformly augmented node GCL as `tau_a -> infinity` | independent softmax limit and numerical convergence | Verified | 2 |
| C3 | node-level TCGA performance | requires full data/training/CV | Not executed | 0 |
| C4 | graph-level TCGA performance | requires full data/training/CV | Not executed | 0 |
| C5 | best result across 13 downstream tasks | requires all baselines and tasks | Not executed | 0 |
| C6 | GraphCL collapse and SupGCL subtype separation | requires pretrained embeddings | Not executed | 0 |
|  | **Prepared** |  |  | **4/12** |

## C1 acceptance contract

1. Verify the factorization and KL chain-rule anchors in pinned paper TeX.
2. Generate independent positive conditional probability tables.
3. Compare direct joint KL against augmentation KL plus supervised
   expectation of node KL.
4. Require maximum absolute error below `1e-12`.
5. Confirm LINCS graph, knockdown metadata, `TeacherSampler`, node InfoNCE,
   and augmentation KL anchors in pinned author code.

## C2 acceptance contract

1. Generate independent teacher/student similarity matrices and node losses.
2. Construct both augmentation distributions with an independent softmax.
3. Increase `tau_a` from 1 to 100000.
4. Require monotonic convergence of both distributions to uniform.
5. Require augmentation KL below `1e-8` and total-loss distance to the
   uniform node-loss average below `1e-4`.
6. Retain the analytic softmax limit as the exact argument; numerical probes
   are finite-precision witnesses.

## Scope guard

The author implementation uses denominator subsampling and stochastic
sampling. The 4/12 result verifies the paper's two mathematical claims and the
declared LINCS data mechanism. It does not certify that the released sampled
loss is an exact unbiased estimator of the full objective.

No points are inferred from paper tables, cached author outputs, leaderboard
entries, or peer reproductions.
