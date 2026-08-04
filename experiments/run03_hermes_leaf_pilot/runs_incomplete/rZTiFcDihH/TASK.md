# Reproduction task — Online Packet Scheduling with Deadlines and Learning

OpenReview: https://openreview.net/forum?id=rZTiFcDihH
Area: Theory
Anchored claims (6):

1. For the deterministic algorithm EDF_Φ^L in 2- and 3-bounded K-OPSD, the Φ-regret upper bound is Õ(√(KT)), matching sleeping-bandit-style rates (Proposition 3.1).

2. The deterministic algorithm ALG^θ achieves competitive ratio θ_K in [√2, Φ) for finite K packet types, breaking the classical golden-ratio Φ = (1+√5)/2 competitive-ratio barrier for 2-bounded deadline instances (Proposition 4.1).

3. The learning variant ALG^θ,U attains a θ_K-regret upper bound of Õ(√(KT)), which is shown to be nearly tight against a matching Ω(√T) lower bound (Theorem 4.2, Theorem 4.3).

4. The randomized algorithm ALG^R2 for 2-bounded instances achieves a 5/4-regret bound of Õ(√(KT)) (Theorem 5.1).

5. The randomized algorithm ALG^Rs for s-bounded (unbounded slackness) instances achieves an e/(e-1)-regret bound of Õ(√(KT)) (Theorem 5.2).

6. Lemma 2.1 establishes that sleeping bandits are a special case of 1-bounded K-OPSD, formally connecting the packet scheduling model to the multi-armed bandit framework (Section 2.2, Lemma 2.1).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
