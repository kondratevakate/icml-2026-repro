# Reproduction task — Understanding Behavior Cloning with Action Quantization

OpenReview: https://openreview.net/forum?id=9uENnRAcSl
Area: Theory
Anchored claims (6):

1. Behavior cloning with quantized actions and log-loss is proven to achieve sample complexity matching known lower bounds, up to the quantization error term (Theorem 2, Section 3.2).

2. Under Probabilistic Incremental Input-to-State Stability (P-IISS) of the dynamics and Relaxed Total Variation Continuity (RTVC) of the expert policy, the regret bound has only polynomial (not exponential) dependence on the horizon H with respect to quantization error epsilon_q (Theorem 3, Definition 3, Definition 4, Section 3.1-3.2).

3. Theorem 6 shows that without a smoothness assumption on the quantizer, non-smooth quantizers can incur regret of order H*Omega(1) even though their in-distribution one-step error is only O(epsilon_q) (Theorem 6, Section 4.1).

4. Theorem 7 proves that model-based data augmentation improves the horizon dependence to H*[sqrt(log|Pi|/n) + epsilon_q] without requiring the policy smoothness (RTVC) assumption (Theorem 7, Section 4.2).

5. Information-theoretic lower bounds (Theorems 8-9) establish that regret must scale at least as H*(1/n + epsilon_q) for deterministic experts and H*(sqrt(1/n) + epsilon_q) for stochastic experts, matching the achievable upper bounds (Section 5, Theorems 8-9).

6. Empirically, binning quantizers are shown to preserve policy smoothness better than learned quantizers, while deterministic experts more often violate the RTVC requirement needed for the sharp regret bound (Section 4.1).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
