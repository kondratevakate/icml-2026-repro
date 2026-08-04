# Reproduction task — Momentum Further Constrains Sharpness at the Edge of Stochastic Stability

OpenReview: https://openreview.net/forum?id=mL4i6z7Miy
Area: Theory
Anchored claims (6):

1. SGD with momentum (SGDM) exhibits an Edge of Stochastic Stability regime whose batch-size-dependent behavior cannot be explained by a single stability threshold, unlike vanilla SGD (Section 3).

2. In the noise-dominated (small-batch) regime, SGDM's mean-square stability condition reduces to a spectral-radius condition rho(I - eta_eff K + eta_eff^2 G) < 1 with effective learning rate eta_eff = eta/(1-beta) (Theorem 4.1).

3. Batch Sharpness converges to a plateau of approximately 2(1-beta)/eta at small batch sizes, indicating momentum drives SGD toward flatter regions than vanilla SGD in this regime (Section 4).

4. Batch Sharpness converges to a plateau of approximately 2(1+beta)/eta for SGDM at large batch sizes, consistent with full-batch gradient descent dynamics and sharper minima (Section 4).

5. Figure 3 empirically shows batch-size-dependent Batch Sharpness plateaus for MLP and CNN architectures on CIFAR-10, with SGD with Nesterov momentum (SGDN) reaching the deterministic plateau at smaller batch sizes than SGDM (Figure 3).

6. Figure 6 shows intervention experiments in which increasing learning rate or momentum, or decreasing batch size, triggers sharp loss 'catapults' once Batch Sharpness exceeds its operating plateau (Figure 6).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
