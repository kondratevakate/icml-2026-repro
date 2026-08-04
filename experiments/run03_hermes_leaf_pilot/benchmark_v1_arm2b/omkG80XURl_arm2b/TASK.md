# Reproduction task — Bridging the Gap Between Average and Discounted TD Learning

OpenReview: https://openreview.net/forum?id=omkG80XURl
Area: Theory
Anchored claims (6):

1. Under i.i.d. sampling in the double-chain formulation, the algorithm converges to a unique, sample-independent fixed point of the projected Bellman equation with sample complexity Õ(ε^-1 η^-2) (Theorem 4.1).

2. Under Markovian sampling with constant stepsizes in the double-chain setting, the same Õ(ε^-1 η^-2) sample complexity is preserved (Theorem 4.2).

3. With decaying stepsizes, the method attains convergence guarantees with no explicit dimension-dependent terms (Theorem 4.3).

4. The paper reduces the condition-number dependence of average-reward TD learning from quartic (prior state of the art) to quadratic, matching the scaling known for discounted TD learning (Section 1, Table 1).

5. A single-chain variant of the algorithm is also analyzed but only attains quartic sample complexity Õ(1/η^4 T) due to decorrelation requirements, contrasting with the double-chain method's quadratic rate (Theorem 4.4).

6. The condition number η1, defined via min over unit vectors of ||Φx||²_Dir + (μᵀΦx)², is shown to satisfy η1 >= (1/2)η3 relative to prior condition-number definitions (Equation 3, Lemma B.3).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
