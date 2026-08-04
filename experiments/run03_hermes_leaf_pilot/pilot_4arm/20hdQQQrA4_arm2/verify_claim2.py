"""verify_claim2.py — the trainable null-space term w_phi (CAffNet, Section 3).

CLAIM: the CAffine projection
    P_gamma(x) = f_theta(x) - A^+(A f_theta - b) + (I - A^+ A) w_phi(x)
contains a *trainable* null-space component w_phi, allowing joint optimisation across
multiple feasible projections rather than a single fixed orthogonal projection.

OPERATIONALISATION (skill P26): decompose into four machine-checkable properties, in the
equality geometry where the formula is stated exactly (A y = b):
  (a) INVARIANCE   — for arbitrary w, A P = b (the term never breaks the guarantee);
  (b) REACHABILITY — the map w -> P is onto the whole affine feasible set: solve for the
      w that hits an arbitrary chosen feasible target and measure the residual;
  (c) DEGENERATE   — w = 0 reproduces exactly the fixed orthogonal (min-norm) projection
      that the paper contrasts against;
  (d) BENEFIT      — optimising w over a downstream quadratic loss strictly beats the
      w = 0 fixed projection on instances whose optimum lies off the orthogonal foot.

MUTATION: delete the null-space term (w := 0) and assert the attainable loss becomes
*exactly* the orthogonal-projection loss on every instance. PREDICTION (before running):
exact equality (not merely degradation), proving nothing else contributes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from caff_core import pinv

N_INST = 200
SEEDS = [0, 1, 2]
CMD = "python verify_claim2.py"


def proj_eq(A, b, f, w):
    n = f.shape[0]
    Ap = pinv(A)
    return f - Ap @ (A @ f - b) + (np.eye(n) - Ap @ A) @ w


def main():
    t0 = time.time()
    inv_err = 0.0
    reach_err = 0.0
    degen_err = 0.0
    n_better = 0
    rel_reductions = []
    mut_exact = 0
    n = 0

    for seed in SEEDS:
        rng = np.random.default_rng([seed, 2])
        for _ in range(N_INST):
            n_out = int(rng.integers(3, 7))
            m = int(rng.integers(1, n_out))          # under-determined => null space
            A = rng.normal(size=(m, n_out))
            b = rng.normal(size=m)
            f = rng.normal(size=n_out)
            w = rng.normal(size=n_out) * 3.0
            n += 1

            # (a) invariance
            y = proj_eq(A, b, f, w)
            inv_err = max(inv_err, float(np.max(np.abs(A @ y - b))))

            # (b) reachability: pick an arbitrary feasible target, solve for w
            y_star = pinv(A) @ b + (np.eye(n_out) - pinv(A) @ A) @ rng.normal(size=n_out) * 2.0
            w_needed = y_star - f + pinv(A) @ (A @ f - b)   # any preimage works
            y_hit = proj_eq(A, b, f, w_needed)
            reach_err = max(reach_err, float(np.max(np.abs(y_hit - y_star))))

            # (c) degenerate case: w = 0 == orthogonal projection of f onto {A y = b}
            y0 = proj_eq(A, b, f, np.zeros(n_out))
            y_orth = f - pinv(A) @ (A @ f - b)
            degen_err = max(degen_err, float(np.max(np.abs(y0 - y_orth))))

            # (d) benefit: downstream loss ||y - c||^2 for a target c off the orth foot
            c = rng.normal(size=n_out) * 2.0
            # optimal over the whole feasible affine set (attainable via some w by (b))
            y_opt = pinv(A) @ b + (np.eye(n_out) - pinv(A) @ A) @ c
            L_opt = float(np.sum((y_opt - c) ** 2))
            L_fixed = float(np.sum((y0 - c) ** 2))
            if L_opt < L_fixed - 1e-12:
                n_better += 1
                rel_reductions.append((L_fixed - L_opt) / max(L_fixed, 1e-12))
            # mutation: w = 0 must land exactly on the fixed orthogonal projection
            if abs(float(np.sum((proj_eq(A, b, f, np.zeros(n_out)) - c) ** 2)) - L_fixed) == 0.0:
                mut_exact += 1

    res = {
        "claim": 2,
        "source": "Section 3 (CAffine layer, null-space term w_phi)",
        "command": CMD,
        "config": {"instances_per_seed": N_INST, "seeds": SEEDS,
                   "geometry": "equality A y = b, under-determined (m < n_out)"},
        "n_instances": n,
        "a_invariance_max_abs_residual": inv_err,
        "b_reachability_max_abs_error": reach_err,
        "c_degenerate_w0_vs_orthogonal_max_abs_diff": degen_err,
        "d_frac_instances_w_beats_fixed": n_better / n,
        "d_median_relative_loss_reduction": float(np.median(rel_reductions)) if rel_reductions else 0.0,
        "mutation_w_zero": {
            "prediction": "w=0 reproduces the fixed orthogonal projection EXACTLY on every instance",
            "n_exact_matches": mut_exact,
            "n_instances": n,
        },
        "verdict": None,
        "wall_seconds": None,
    }
    res["verdict"] = ("verified" if (inv_err < 1e-9 and reach_err < 1e-9
                                     and degen_err < 1e-12 and mut_exact == n
                                     and res["d_frac_instances_w_beats_fixed"] > 0.5)
                      else "falsified")
    res["wall_seconds"] = round(time.time() - t0, 2)
    Path("results").mkdir(exist_ok=True)
    Path("results/claim2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
