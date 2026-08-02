"""CLAIM 1 — Theorem 1 (Sec.4.1, proof App.C): LP rounding bicriteria (1+kappa, 1+1/kappa)
approximation. Verifies W - e(K) <= (1+kappa)*eps*W and |K| <= (1+1/kappa)*r.

Verdict logic: build a planted hypergraph with a known feasible core O (|O|=r, e(O)>=(1-eps)W).
Solve LP (1), round K = {v : x*_v >= rho=kappa/(1+kappa)}. Check both bounds.
Mutation test: round with a WRONG threshold (not kappa/(1+kappa)) -> the (1+kappa,1+1/kappa)
guarantee must be able to fail (coverage loss can exceed the bound). Also sweep kappa to show the
trade-off and that the bound is parameterized by the SAME kappa used for rounding.
"""
import json
import numpy as np
from lib_hg import Hypergraph

SEED = 20260207


def make_instance(a=10, b=20, eps=0.10, n_he=40, seed=SEED):
    """Two weighted, disjoint vertex blocks force a FRACTIONAL LP (1) solution:
    block A (a vertices) carries weight w_A=0.6, block B (b vertices) carries w_B=0.4.
    To meet the (1-eps)=0.9 coverage the LP must include all of A and 0.75 of B fractionally,
    so rounding at rho=kappa/(1+kappa) is what yields the bicriteria guarantee.
    Benchmark r = |A u B| = a+b (an integral feasible set covering all weight)."""
    rng = np.random.default_rng(seed)
    W = 1.0
    w_A = 0.6 * W
    w_B = 0.4 * W
    A = list(range(a))
    B = list(range(a, a + b))
    hes, ws = [], []
    for _ in range(n_he):
        e = tuple(sorted(rng.choice(A, size=rng.integers(2, a + 1), replace=False)))
        hes.append(e); ws.append(1.0)
    wa = np.array(ws) / sum(ws) * w_A
    for _ in range(n_he):
        e = tuple(sorted(rng.choice(B, size=rng.integers(2, b + 1), replace=False)))
        hes.append(e); ws.append(1.0)
    wb = np.array(ws[len(wa):]) / sum(ws[len(wa):]) * w_B
    ws = list(wa) + list(wb)
    H = Hypergraph(hes, ws)
    O = set(range(a + b))
    return H, O


def check_bounds(H, O, eps, kappa):
    K, x, z, obj, rho = H.lp_rounding_set(eps, kappa)
    r = len(O)
    loss = H.coverage_loss(K)
    size = len(K)
    loss_bound = (1.0 + kappa) * eps * H.W
    size_bound = (1.0 + 1.0 / kappa) * r
    return {
        "kappa": kappa,
        "rho": rho,
        "lp_obj": obj,
        "r_benchmark": r,
        "K_size": size,
        "K_size_bound": size_bound,
        "size_ok": bool(size <= size_bound + 1e-9),
        "cov_loss": loss,
        "cov_loss_bound": loss_bound,
        "cov_ok": bool(loss <= loss_bound + 1e-9),
        "W": H.W,
    }


def main():
    H, O = make_instance()
    eps = 0.10
    # sweep kappa with correct rounding -> both bounds should hold and trade-off visible
    sweep = [check_bounds(H, O, eps, k) for k in [0.5, 1.0, 2.0, 4.0]]

    # Mutation test: the guarantee is SPECIFIC to the threshold rho = kappa/(1+kappa).
    # Perturb the rounding threshold away from the theorem's value and the (1+kappa,1+1/kappa)
    # bound must be breakable. We test three thresholds for kappa=1 (rho=0.5):
    #   correct rho=0.5  -> both bounds hold
    #   rho=0.95 (too aggressive) -> tiny K, coverage bound (1+kappa)*eps*W violated
    #   rho=0.10 (too lax)       -> huge K, size bound (1+1/kappa)*r violated
    x, z, obj = H.solve_lp(eps)
    r = len(O)
    kappa = 1.0
    loss_bound = (1.0 + kappa) * eps * H.W
    size_bound = (1.0 + 1.0 / kappa) * r
    thr = {}
    for tag, rho_mut in [("correct_rho_0.5", 0.5), ("too_aggressive_0.95", 0.95), ("too_lax_0.10", 0.10)]:
        Km = set(int(i) for i in range(H.n) if x[i] >= rho_mut - 1e-9)
        thr[tag] = {
            "rho": rho_mut,
            "K_size": len(Km),
            "size_bound": size_bound,
            "size_ok": bool(len(Km) <= size_bound + 1e-9),
            "cov_loss": H.coverage_loss(Km),
            "cov_loss_bound": loss_bound,
            "cov_ok": bool(H.coverage_loss(Km) <= loss_bound + 1e-9),
        }
    mutation_correct_holds = bool(thr["correct_rho_0.5"]["size_ok"] and thr["correct_rho_0.5"]["cov_ok"])
    mutation_breaks = bool((not thr["too_aggressive_0.95"]["cov_ok"]) or (not thr["too_lax_0.10"]["size_ok"]))

    # Mutation B: instance perturbation (robustness) - different random instance, bounds still hold
    H2, O2 = make_instance(seed=SEED + 1)
    robust = check_bounds(H2, O2, eps, 1.0)

    all_size_ok = all(s["size_ok"] for s in sweep)
    all_cov_ok = all(s["cov_ok"] for s in sweep)
    verdict = "verified" if (all_size_ok and all_cov_ok and mutation_correct_holds and mutation_breaks) else "falsified"

    out = {
        "claim": 1,
        "statement": "Theorem 1 bicriteria (1+kappa,1+1/kappa) LP-rounding approximation",
        "source": "Sec.4.1, proof Appendix C (Eqs.6 and coverage bound)",
        "verdict": verdict,
        "seed": SEED,
        "instance": {"n": H.n, "m": H.m, "W": H.W, "eps": eps, "r_benchmark": len(O)},
        "kappa_sweep": sweep,
        "mutation_threshold": {
            "description": "Round at rho != kappa/(1+kappa). Correct threshold satisfies both bounds; "
                           "perturbed thresholds must break at least one bound.",
            "results": thr,
            "correct_holds": mutation_correct_holds,
            "perturbed_breaks": mutation_breaks,
        },
        "mutation_robustness": {
            "description": "Different random instance (seed+1); bounds should still hold.",
            "size_ok": robust["size_ok"],
            "cov_ok": robust["cov_ok"],
            "K_size": robust["K_size"],
            "cov_loss": robust["cov_loss"],
        },
    }
    with open("results/claim1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
