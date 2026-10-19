# Bayesian causal meta-learning claim decomposition

Paper: **Bayesian Meta-Learning with Expert Feedback for Task-Shift Adaptation through Causal Embeddings**

OpenReview ID: `k76ll7aQyE`  
arXiv: `2602.19788v1`

Each claim is worth two points.

1. **Embedding-conditional prior.** The release implements the paper's Gaussian task prior with mean `theta + W z_t`.
2. **Prior-risk continuity and decomposition.** The bounded-loss Lipschitz proposition and the expert/causal/OOD error decomposition are correct under their stated assumptions.
3. **Negative-transfer theorem.** The sufficient condition in Theorem 5 follows from the stated assumptions with a valid constant and matches the method actually released.
4. **Synthetic task-shift evidence.** The released toy data and code independently support reduced negative transfer under task shift and expert-guided embedding inference.
5. **Clinical cross-disease evidence.** The released artifacts independently support the UK Biobank cross-disease results.

