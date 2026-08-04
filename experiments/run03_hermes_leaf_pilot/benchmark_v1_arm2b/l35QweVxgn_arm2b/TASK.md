# Reproduction task — On the Theory of Continual Learning with Gradient Descent for Neural Networks

OpenReview: https://openreview.net/forum?id=l35QweVxgn
Area: Theory
Anchored claims (6):

1. Theorem 1 gives a closed-form bound on train-time forgetting for task k after training on K-k subsequent tasks, of order O~(ηT√(K-k)/(d√n) + ηT√(K-k)/(d²polylog(d)) + η²T²K²/√m) (Theorem 1).

2. The forgetting bounds hold under the parameter regime n = Θ~(d²K) samples, m = Ω~(d⁸K⁴) hidden-layer width, and ηT = Θ(d²) training iterations on a d-dimensional XOR cluster dataset with K tasks (Theorem 1).

3. Theorem 2 shows that under the same width/sample/iteration conditions as Theorem 1, the misclassification error remains uniformly small across all K tasks with high probability after KT gradient descent iterations (Theorem 2).

4. Theorem 3 bounds the delayed generalization gap by η T exp(ηT(K-k+1)/√m) / n for Lipschitz, smooth loss functions, showing the gap decays with sample size n (Theorem 3).

5. Theorem 4 gives an improved generalization gap bound for self-bounded losses that scales poly-logarithmically rather than linearly in the number of iterations T, depending on the cumulative training loss of later tasks (Theorem 4).

6. Test-time forgetting is decomposed as the sum of train-time forgetting (Theorem 1) and the delayed generalization gap (Theorem 3/4), showing that network width, sample size, and later tasks' data jointly, not individually, control forgetting (Theorem 1, Theorem 3).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
