"""CLAIM 2 - Theorem 2 (Sec.4.2, proof App.D): monotonicity K_{tau1} <= K_{tau2} for tau1<tau2,
derived from parametric min-cuts, and the parametric algorithm equals the LP-rounding algorithm.
Also reproduces Lemma 2 (nested sequence S_0..S_k with k<=n).

Reproduction:
  * Build a multi-block hypergraph so the conformal subgraph has several nested levels.
  * Run the parametric min-cut network D_lambda (Sec.4.2) over a lambda grid -> nested sequence.
  * Verify the sequence is nested and |sequence| <= n+1 (Lemma 2).
  * For the chosen kappa, eps, compute K^tau = argmin_{S in S_1..S_k} {|S| : W-e(S) <= (1+kappa)*eps*W}
    and verify K^tau equals the LP-rounding set K (Theorem 2 equality).
  * Verify nesting of K^tau across a grid of tau.
Mutation: the OPTIMAL (unrounded) cover is NOT monotone -- reproduce Example 1 of the paper and
show the optimal solution fails nesting, while the LP/parametric rounding is monotone.
"""
import json
import numpy as np
from lib_hg import Hypergraph
from lib_flow import parametric_sequence, min_cut_K

SEED = 20260207


def multi_block_instance(block_weights=(0.4, 0.3, 0.2, 0.1), per_block=30, seed=SEED):
    rng = np.random.default_rng(seed)
    W = 1.0
    hes, ws = [], []
    vstart = 0
    blocks = []
    for bi, bw in enumerate(block_weights):
        a = 6 + bi * 2  # blocks of growing size
        A = list(range(vstart, vstart + a))
        blocks.append(set(A))
        local = []
        for _ in range(per_block):
            e = tuple(sorted(rng.choice(A, size=rng.integers(2, a + 1), replace=False)))
            hes.append(e); ws.append(1.0); local.append(1.0)
        wa = np.array(local) / sum(local) * (bw * W)
        for i in range(len(local)):
            ws[len(ws) - len(local) + i] = wa[i]
        vstart += a
    return Hypergraph(hes, ws), blocks


def nested_sequence_K(H, kappa, eps, lam_grid=None):
    if lam_grid is None:
        lam_grid = np.logspace(-4, 4, 240)
    seq = parametric_sequence(H.V, H.hyperedges, H.weights, lam_grid)
    seq = [set(H.vid[v] for v in S) for S in seq]  # map global ids -> indices
    # ensure empty set first and drop empties other than S0
    seq = [s for s in seq]
    return seq


def K_tau_from_seq(H, seq, kappa, eps):
    allowed = (1.0 + kappa) * eps * H.W
    # smallest S_j (j>=1) with loss <= allowed
    best = None
    for S in seq:
        if len(S) == 0:
            continue
        if H.coverage_loss(S) <= allowed + 1e-9:
            if best is None or len(S) < len(best):
                best = S
    return best


def main():
    H, blocks = multi_block_instance()
    eps = 0.10
    kappa = 1.0
    seq = nested_sequence_K(H, kappa, eps)
    # nesting check
    nested = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    k = len(seq) - 1
    k_le_n = bool(k <= H.n)

    # Theorem 2 equality: parametric K^tau vs LP-rounding K
    Kpar = K_tau_from_seq(H, seq, kappa, eps)
    Klp, x, z, obj, rho = H.lp_rounding_set(eps, kappa)
    equal = (Kpar == Klp)

    # nesting across tau grid (tau = 1-eps_target)
    tau_grid = [0.55, 0.65, 0.75, 0.85, 0.92]
    Ks = []
    for tau in tau_grid:
        epst = 1.0 - tau
        Ks.append(K_tau_from_seq(H, seq, kappa, epst))
    nesting_tau = all(Ks[i] <= Ks[i + 1] for i in range(len(Ks) - 1))

    # ---- Mutation: optimal (unrounded) cover is NOT monotone (Example 1 of paper) ----
    # 3 disjoint paths P1,P2 (2 edges each, w=0.3) and P3 (3 edges, w=0.4).
    # Optimal cover for target tau = min-size vertex set covering >= tau*W.
    P1, P2, P3 = (0, 1), (2, 3), (4, 5, 6)
    hes = [P1, P2, P3]
    ws = [0.3, 0.3, 0.4]
    W = 1.0

    def optimal_cover(tau):
        target = tau * W
        best = None
        # brute force over nonempty subsets of the 3 paths (2^3-1)
        for mask in range(1, 8):
            sel = [hes[i] for i in range(3) if mask & (1 << i)]
            cov = sum(ws[i] for i in range(3) if mask & (1 << i))
            if cov >= target - 1e-9:
                S = set()
                for e in sel:
                    S |= set(e)
                if best is None or len(S) < len(best):
                    best = S
        return best

    K_opt_06 = optimal_cover(0.6)
    K_opt_07 = optimal_cover(0.7)
    opt_nested = bool(K_opt_06 <= K_opt_07)
    # show the LP/parametric rounding IS monotone on the same example
    Hex = Hypergraph(hes, ws)
    seq_ex = [set(Hex.vid[v] for v in S) for S in
              parametric_sequence(Hex.V, Hex.hyperedges, Hex.weights, np.logspace(-4, 4, 200))]
    Kp_06 = K_tau_from_seq(Hex, seq_ex, 1.0, 1.0 - 0.6)
    Kp_07 = K_tau_from_seq(Hex, seq_ex, 1.0, 1.0 - 0.7)
    lp_monotone = bool(Kp_06 <= Kp_07)

    verdict = "verified" if (nested and k_le_n and equal and nesting_tau and (not opt_nested) and lp_monotone) else "falsified"

    out = {
        "claim": 2,
        "statement": "Theorem 2 monotonicity (nested K_tau) + equality of parametric & LP-rounding algorithms (Lemma 2 / Sec.4.2)",
        "source": "Sec.4.2, proof Appendix D; Lemma 2 (Gallo-Grigoriadis-Tarjan parametric min-cut)",
        "verdict": verdict,
        "seed": SEED,
        "instance": {"n": H.n, "m": H.m, "W": H.W, "eps": eps, "kappa": kappa},
        "parametric_sequence": {
            "num_levels_k": k,
            "n": H.n,
            "k_le_n": k_le_n,
            "nested": nested,
            "sizes": [len(s) for s in seq],
        },
        "theorem2_equality": {"K_parametric_size": len(Kpar) if Kpar else 0,
                              "K_lp_size": len(Klp),
                              "equal": equal},
        "nesting_across_tau": {"tau_grid": tau_grid,
                               "sizes": [len(s) if s else 0 for s in Ks],
                               "nested": nesting_tau},
        "mutation_optimal_not_monotone": {
            "description": "Example 1 of paper: 3 disjoint paths (0.3,0.3,0.4). Optimal (unrounded) "
                           "cover is NOT nested; LP/parametric rounding IS nested.",
            "K_opt_tau0.6": sorted(K_opt_06) if K_opt_06 else [],
            "K_opt_tau0.7": sorted(K_opt_07) if K_opt_07 else [],
            "optimal_nested": opt_nested,
            "K_lp_tau0.6": sorted(Kp_06) if Kp_06 else [],
            "K_lp_tau0.7": sorted(Kp_07) if Kp_07 else [],
            "lp_rounding_nested": lp_monotone,
        },
    }
    with open("results/claim2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
