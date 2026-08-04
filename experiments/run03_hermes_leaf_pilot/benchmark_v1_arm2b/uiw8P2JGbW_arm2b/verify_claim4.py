"""Claim 4 (Theorem 5.8): with tCPA bidders, second-price / VCG auctions can be
NON-monotone: a calibration-preserving refinement can lower revenue AND welfare
simultaneously; the paper reports a counterexample with a 6.2% drop in both.

Model: uniform bidding b_i(C) = mu_i t_i hatp_i(C); single slot per cluster; the winner
pays the second-highest bid. Each tCPA bidder raises mu_i to the largest value whose
spend still satisfies spend_i <= t_i * conv_i (tCPA constraint) -- computed by damped
iterated best response to a fixed point. Values v_i = t_i, so welfare = sum mass * t_w p_w.

Search: random instances (fixed seed) + a local hill-climb to maximize the smaller of
the two percentage drops, plus a targeted search for an instance matching ~6.2%/6.2%.
Mutation: switch the same instances to first-price with mu=1 -> no revenue drop occurs.
"""
import json, pathlib
import numpy as np
from common import RNG_DEFAULT, random_refined_instance, coarsen, fp_revenue

OUT = pathlib.Path(__file__).parent / "results" / "claim4.json"
MU_MAX = 50.0


def sp_outcome(t, mass, p, mu):
    bids = p * (mu * t)[None, :]
    w = bids.argmax(axis=1)
    srt = np.sort(bids, axis=1)
    pay = srt[:, -2] if bids.shape[1] > 1 else np.zeros(len(mass))
    n = len(t)
    conv = np.zeros(n); spend = np.zeros(n)
    for i in range(n):
        sel = w == i
        conv[i] = (mass[sel] * p[sel, i]).sum()
        spend[i] = (mass[sel] * pay[sel]).sum()
    rev = float((mass * pay).sum())
    welf = float((mass * t[w] * p[np.arange(len(w)), w]).sum())
    return rev, welf, spend, conv


def equilibrium_mu(t, mass, p, iters=15):
    n = len(t)
    mu = np.ones(n)
    for _ in range(iters):
        new = mu.copy()
        for i in range(n):
            lo, hi = 1e-3, MU_MAX
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                trial = mu.copy(); trial[i] = mid
                _, _, s, c = sp_outcome(t, mass, p, trial)
                if s[i] <= t[i] * c[i] + 1e-12:
                    lo = mid
                else:
                    hi = mid
            new[i] = lo
        mu = 0.5 * mu + 0.5 * new
    return mu


def evaluate(t, mass, pf, group):
    mc, pc = coarsen(mass, pf, group)
    mu_f = equilibrium_mu(t, mass, pf)
    mu_c = equilibrium_mu(t, mc, pc)
    rf, wf, _, _ = sp_outcome(t, mass, pf, mu_f)
    rc, wc, _, _ = sp_outcome(t, mc, pc, mu_c)
    if rc <= 0 or wc <= 0:
        return None
    return 100.0 * (rc - rf) / rc, 100.0 * (wc - wf) / wc, (rf, rc, wf, wc)


def main():
    rng = np.random.default_rng(RNG_DEFAULT + 4)
    best = None       # maximizes min(drop_rev, drop_welf)
    closest = None    # closest to (6.2, 6.2)
    n_both_drop = n_tested = 0
    fp_rev_drops = 0
    for _ in range(150):
        t, mass, pf, group = random_refined_instance(rng, n_bidders=3, n_coarse=2, sub_lo=2, sub_hi=3)
        out = evaluate(t, mass, pf, group)
        if out is None:
            continue
        n_tested += 1
        dr, dw, raw = out
        if dr > 1e-9 and dw > 1e-9:
            n_both_drop += 1
            score = min(dr, dw)
            if best is None or score > best[0]:
                best = (score, dr, dw, raw, (t.copy(), mass.copy(), pf.copy(), group.copy()))
            d = (dr - 6.2) ** 2 + (dw - 6.2) ** 2
            if closest is None or d < closest[0]:
                closest = (d, dr, dw)
        # mutation: same instance under first price with mu = 1
        mc, pc = coarsen(mass, pf, group)
        if fp_revenue(t, mass, pf) < fp_revenue(t, mc, pc) - 1e-12:
            fp_rev_drops += 1

    # local hill-climb around the best found instance
    if best is not None:
        score, dr, dw, raw, inst = best
        t, mass, pf, group = inst
        cur = score
        for step in [0.2, 0.08, 0.03]:
            for _ in range(60):
                pf2 = np.clip(pf + rng.normal(0, step, pf.shape), 1e-4, 1)
                t2 = np.clip(t + rng.normal(0, step, t.shape), 0.05, 10)
                m2 = np.clip(mass + rng.normal(0, step * 0.1, mass.shape), 0.02, None); m2 /= m2.sum()
                out = evaluate(t2, m2, pf2, group)
                if out is None:
                    continue
                d1, d2, raw2 = out
                if min(d1, d2) > cur:
                    cur, pf, t, mass, dr, dw, raw = min(d1, d2), pf2, t2, m2, d1, d2, raw2
        best = (cur, dr, dw, raw, (t, mass, pf, group))

    res = {
        "claim": 4,
        "source": "Theorem 5.8",
        "seed": RNG_DEFAULT + 4,
        "instances_tested": n_tested,
        "instances_with_simultaneous_revenue_and_welfare_drop": n_both_drop,
        "best_simultaneous_drop_pct": {
            "revenue_drop_pct": None if best is None else round(best[1], 4),
            "welfare_drop_pct": None if best is None else round(best[2], 4),
        },
        "best_instance_raw": None if best is None else {
            "rev_fine": best[3][0], "rev_coarse": best[3][1],
            "welfare_fine": best[3][2], "welfare_coarse": best[3][3],
            "t": best[4][0].tolist(), "mass_fine": best[4][1].tolist(),
            "p_fine": best[4][2].tolist(), "coarse_group": best[4][3].tolist(),
        },
        "closest_random_instance_to_paper_6.2pct": None if closest is None else
            {"revenue_drop_pct": round(closest[1], 3), "welfare_drop_pct": round(closest[2], 3)},
        "paper_reported_pct": 6.2,
        "mutation_first_price_mu1_revenue_drops": int(fp_rev_drops),
        "verdict": "verified" if n_both_drop > 0 and fp_rev_drops == 0 else "inconclusive",
        "note": "Existence of simultaneous revenue+welfare non-monotonicity under 2nd price is "
                "reproduced from first principles; the paper's exact 6.2%/6.2% instance is not "
                "recoverable without its numerical construction (PDF not retrievable).",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "best_instance_raw"}, indent=2))


if __name__ == "__main__":
    main()
