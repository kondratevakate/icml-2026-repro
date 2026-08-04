"""Claim 3 (Theorem 5.2 + Corollary 5.3): in FPA with tCPA bidders (no budgets),
uniform bidding at mu = 1 is simultaneously
  (i)  conversion-maximizing for every bidder (best response within the tCPA constraint),
  (ii) revenue-maximizing among feasible (tCPA-satisfying) uniform-bidding profiles,
  (iii) welfare-maximizing when v_i = t_i;
and (Cor. 5.3) welfare is then monotone under model refinement.

Feasibility: spend_i = sum_C mass_C * mu_i t_i p_i(C) * 1[i wins C] = mu_i t_i * conv_i,
so the tCPA constraint spend_i <= t_i * conv_i holds iff mu_i <= 1 (conv_i > 0).

Mutation: set v_i != t_i -> the mu=1 allocation stops being welfare-maximal.
"""
import json, pathlib
import numpy as np
from common import RNG_DEFAULT, random_refined_instance, coarsen, fp_revenue, fp_alloc, welfare

OUT = pathlib.Path(__file__).parent / "results" / "claim3.json"
N_INST, N_MU = 2000, 300


def spend_conv(t, mass, p, mu):
    w = fp_alloc(t, mass, p, mu)
    idx = np.arange(len(w))
    n = len(t)
    conv = np.zeros(n); spend = np.zeros(n)
    for i in range(n):
        sel = w == i
        conv[i] = (mass[sel] * p[sel, i]).sum()
        spend[i] = (mass[sel] * mu[i] * t[i] * p[sel, i]).sum()
    return spend, conv, w


def main():
    rng = np.random.default_rng(RNG_DEFAULT + 3)
    feas_mismatch = rev_viol = conv_viol = welf_viol = 0
    welf_mono_viol = 0
    mut_welf_beaten = 0
    for _ in range(N_INST):
        t, mass, pf, group = random_refined_instance(rng)
        n = len(t)
        ones = np.ones(n)
        s1, c1, w1 = spend_conv(t, mass, pf, ones)
        rev1 = fp_revenue(t, mass, pf, ones)
        welf1 = welfare(t, mass, pf, w1)           # v = t
        v_alt = t * rng.uniform(0.3, 3.0, size=n)  # mutation values
        welf1_alt = welfare(v_alt, mass, pf, w1)
        # brute-force welfare optimum (per-cluster argmax of v_i p_i)
        opt_t = float((mass * (pf * t).max(axis=1)).sum())
        opt_alt = float((mass * (pf * v_alt).max(axis=1)).sum())
        if welf1 < opt_t - 1e-12:
            welf_viol += 1
        if welf1_alt < opt_alt - 1e-9:
            mut_welf_beaten += 1
        for _ in range(N_MU):
            mu = rng.uniform(0.2, 1.6, size=n)
            s, c, w = spend_conv(t, mass, pf, mu)
            feasible = np.all(s <= t * c + 1e-12)
            if feasible != bool(np.all(mu <= 1 + 1e-12) or np.all(c[mu > 1] == 0)):
                feas_mismatch += 1
            if feasible:
                if fp_revenue(t, mass, pf, mu) > rev1 + 1e-12:
                    rev_viol += 1
                pass
                if welfare(t, mass, pf, w) > welf1 + 1e-12:
                    welf_viol += 1
        # unilateral deviations: bidder i deviates, others stay at mu = 1
        for i in range(n):
            for m in [0.2, 0.5, 0.8, 0.95, 1.05, 1.3, 1.6]:
                mu = ones.copy(); mu[i] = m
                s_, c_, _ = spend_conv(t, mass, pf, mu)
                if s_[i] <= t[i] * c_[i] + 1e-12 and c_[i] > c1[i] + 1e-9:
                    conv_viol += 1
        # Corollary 5.3 welfare monotonicity under refinement (v = t, mu = 1)
        mc, pc = coarsen(mass, pf, group)
        wc = fp_alloc(t, mc, pc)
        if welfare(t, mass, pf, w1) < welfare(t, mc, pc, wc) - 1e-12:
            welf_mono_viol += 1

    res = {
        "claim": 3,
        "source": "Theorem 5.2, Corollary 5.3",
        "seed": RNG_DEFAULT + 3,
        "n_instances": N_INST,
        "mu_profiles_per_instance": N_MU,
        "feasibility_iff_mu_le_1_mismatches": int(feas_mismatch),
        "feasible_profiles_beating_mu1_on_revenue": int(rev_viol),
        "unilateral_deviations_beating_mu1_on_own_conversions": int(conv_viol),
        "feasible_profiles_beating_mu1_on_welfare_v_eq_t": int(welf_viol),
        "corollary53_welfare_monotonicity_violations": int(welf_mono_viol),
        "mutation_v_ne_t_instances_where_mu1_not_welfare_optimal": int(mut_welf_beaten),
        "verdict": "verified" if (feas_mismatch == 0 and rev_viol == 0 and conv_viol == 0
                                  and welf_viol == 0 and welf_mono_viol == 0
                                  and mut_welf_beaten > 0) else "inconclusive",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
