# Reproduction task — Context-free Recognition with Transformers

OpenReview: https://openreview.net/forum?id=JOyxs9ElI7
Area: Theory
Anchored claims (6):

1. Theorem 3.1 proves that all context-free languages can be recognized by looped transformers using O(log(n)) looping layers and O(n^6) padding tokens (Section 3, Theorem 3.1).

2. Theorem 4.1 shows unambiguous context-free languages can be recognized with fewer resources than the general case: O(log^2(n)) looping layers and O(n^3) padding tokens (Section 4, Theorem 4.1).

3. Theorem 4.2 further reduces requirements for unambiguous linear context-free languages to O(log(n)) looping layers and O(n^2) padding tokens by exploiting the linearity constraint (Section 4.2, Theorem 4.2).

4. Lemma 4.1 shows that Boolean formula evaluation can be computed in O(log(n)) steps via a parallel pebbling argument, which underlies the padding-token constructions used in the main theorems (Section 4, Lemma 4.1).

5. Corollary 4.1 shows that Boolean Formula Value Problem (BFVP) recognition requires zero padding tokens while retaining logarithmic depth (Section 4, Corollary 4.1).

6. Table 1 summarizes the padding-versus-depth tradeoff across general, unambiguous, and linear-unambiguous context-free language classes, showing that grammatical restrictions systematically reduce required padding (Table 1).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
