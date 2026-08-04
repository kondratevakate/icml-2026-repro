"""verify_claim1.py — Theorem 3.5 (Universal Approximation of CAffNet).

Claim: if F_theta universally approximates F, then F_CAffNet approximates
F_target = {f_t in F | A(x) f_t <= b(x)}, with ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K
where K = eps / (3 + 3 sqrt(n_out)).  (Theorem 3.5, proof in Appendix C.)

What is actually tested (per instance x, exhaustive grid over n_out, m, p,
rank pattern, seed):
  hypotheses of the proof: f_t feasible, ||f_theta - f_t||_p < K, ||w_phi||_p < 2K
  conclusions tested:
    (C1) ||P* - f_t||_p  < (3 + 3 sqrt(n_out)) K      (the anchored bound)
    (C2) ||P* - f_th||_p < (2 + 3 sqrt(n_out)) K      (intermediate)
    (C3) exists gamma with ||f_t - P_gamma||_p < (1 + 3 sqrt(n_out)) K
    (C4) ||A_g^dag A_g||_p <= sqrt(n_out) and ||I - A_g^dag A_g||_p <= sqrt(n_out)  (Eq. 31)
    (C5) a feasible candidate always exists (Lemma 3.3 -> Thm 3.4)
  Because eps = (3+3 sqrt(n_out)) K, (C1) is exactly "error < eps": the UAT conclusion.

MUTATION TESTS (must break the bound):
  M1: violate premise (28) — draw ||w_phi||_p = 50 K instead of < 2K.
  M2: replace argmin in Eq. (12) by argmax (choose the FARTHEST feasible candidate).
Both are expected to produce (C1) violations; if they did not, the observed
bound would not be evidence about the mechanism.

Run: .venv/bin/python verify_claim1.py
"""
import itertools
import json
import sys
import time

import numpy as np

from caffnet_core import P_gamma, caffnet_output, gamma_set, random_feasible_system

P_LIST = [1.0, 1.5, 2.0, 3.0]
N_OUT_LIST = [1, 2, 3, 4]
M_EXTRA = [0, 1, 3]          # m = n_out + extra  (m can exceed n_out)
DEP_LIST = [0, 1, 2]         # number of linearly dependent rows forced
SEEDS = list(range(25))      # exhaustive over this seed grid, never a single seed
K = 0.01


def pnorm(v, p):
    return float(np.linalg.norm(v, ord=p))


def sample_instance(rng, n, m, dep, p, boundary=False):
    """Return A, b, f_t (feasible), f_theta with ||f_theta-f_t||_p < K.

    boundary=True pushes f_t onto an active constraint so that roughly half of the
    K-perturbations f_theta are INFEASIBLE, i.e. Case 2 of the Appendix-C proof
    (the branch where the CAffine projection is actually invoked).
    """
    for _ in range(200):
        A, b, y0 = random_feasible_system(rng, m, n, dep_rows=dep)
        if boundary:
            j = int(rng.integers(0, m))
            aj = A[j]
            nn2 = float(aj @ aj)
            if nn2 < 1e-12:
                continue
            y0 = y0 + (b[j] - aj @ y0) * aj / nn2      # make constraint j active
            if not np.all(A @ y0 <= b + 1e-12):
                continue
            d = rng.normal(size=n)
            nd = pnorm(d, p)
            if nd == 0:
                continue
            d = d / nd * (0.99 * K * rng.uniform(0.2, 1.0))
            return A, b, y0, y0 + d
        # f_t must be strictly feasible so the target class F_target is non-empty here
        if np.all(A @ y0 <= b - 1e-9):
            d = rng.normal(size=n)
            nd = pnorm(d, p)
            if nd == 0:
                continue
            d = d / nd * (0.99 * K * rng.uniform(0.2, 1.0))
            return A, b, y0, y0 + d
    return None


def run(mode):
    """mode in {'faithful','M1_big_w','M2_argmax'}"""
    rng_master = np.random.default_rng(20260523)
    rows = []
    worst = 0.0
    n_viol = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}
    n_cases = 0
    n_case2 = 0
    for n_out, extra, dep, p, seed in itertools.product(
            N_OUT_LIST, M_EXTRA, DEP_LIST, P_LIST, SEEDS):
        m = n_out + extra
        if dep >= m:
            continue
        rng = np.random.default_rng(
            [int(rng_master.integers(1 << 30)), n_out, m, dep, int(p * 10), seed])
        inst = sample_instance(rng, n_out, m, dep, p, boundary=(seed % 2 == 1))
        if inst is None:
            continue
        A, b, f_t, f_th = inst
        sq = np.sqrt(n_out)
        eps = (3 + 3 * sq) * K

        wdir = rng.normal(size=n_out)
        wdir = wdir / max(pnorm(wdir, p), 1e-12)
        w = wdir * (1.9 * K * rng.uniform(0.1, 1.0))
        if mode == "M1_big_w":
            w = wdir * (50.0 * K)
        select = "max" if mode == "M2_argmax" else "min"

        y, info = caffnet_output(f_th, w, A, b, p=p, select=select)
        n_cases += 1
        if y is None:
            n_viol["C5"] += 1
            continue
        e1 = pnorm(y - f_t, p)
        e2 = pnorm(y - f_th, p)
        if not (e1 < eps):
            n_viol["C1"] += 1
        if not (e2 < (2 + 3 * sq) * K):
            n_viol["C2"] += 1
        # C3: some gamma achieves the (1+3 sqrt n) K bound.
        # NOTE: Appendix C derives this only in Case 2 (f_theta infeasible), where the
        # segment f_theta -> f_t crosses a face; in Case 1 the projection is never
        # invoked, so the test is restricted to Case 2 instances.
        if info["case"] == "projected":
            n_case2 += 1
            best = min(pnorm(P_gamma(f_th, w, A, b, g) - f_t, p)
                       for g in gamma_set(m, n_out))
            if not (best < (1 + 3 * sq) * K):
                n_viol["C3"] += 1
        # C4: Eq. (31) matrix-norm bounds
        for g in gamma_set(m, n_out):
            Ag = A[list(g), :]
            Agd = np.linalg.pinv(Ag)
            Pm = Agd @ Ag
            I = np.eye(n_out)
            npn = 1 if p == 1 else (2 if p == 2 else None)
            # induced p-norm: exact for p=1,2,inf; use power iteration proxy via
            # sampling for general p
            def induced(Mx):
                if npn is not None:
                    return float(np.linalg.norm(Mx, ord=npn))
                vs = rng.normal(size=(200, n_out))
                num = np.linalg.norm(vs @ Mx.T, ord=p, axis=1)
                den = np.linalg.norm(vs, ord=p, axis=1)
                return float(np.max(num / den))
            if induced(Pm) > sq + 1e-8 or induced(I - Pm) > sq + 1e-8:
                n_viol["C4"] += 1
                break
        worst = max(worst, e1 / eps)
        rows.append(dict(n_out=n_out, m=m, dep=dep, p=p, seed=seed,
                         err_to_ft=e1, eps=eps, ratio=e1 / eps,
                         n_feasible=info["n_feasible"], case=info["case"]))
    return dict(mode=mode, n_cases=n_cases, violations=n_viol,
                n_case2_projected=n_case2,
                worst_ratio_err_over_eps=worst, n_records=len(rows))


if __name__ == "__main__":
    t0 = time.time()
    out = {"claim": 1,
           "claim_text": "Theorem 3.5: CAffNet inherits universal approximation, "
                         "||P* - f_t||_p < (3 + 3 sqrt(n_out)) K = eps",
           "source": "Theorem 3.5 (Sec. 3.2), proof Appendix C, Eqs. (23),(28),(31)",
           "K": K,
           "grid": {"n_out": N_OUT_LIST, "m_extra": M_EXTRA, "dependent_rows": DEP_LIST,
                    "p": P_LIST, "seeds": SEEDS},
           "command": ".venv/bin/python verify_claim1.py"}
    out["faithful"] = run("faithful")
    out["mutation_M1_big_w"] = run("M1_big_w")
    out["mutation_M2_argmax"] = run("M2_argmax")
    out["elapsed_s"] = time.time() - t0
    json.dump(out, open("results/claim1.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("faithful", "mutation_M1_big_w", "mutation_M2_argmax",
                               "elapsed_s")}, indent=2))
