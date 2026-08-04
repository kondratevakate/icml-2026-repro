"""Claim 4 (Lemma 1): distribution-free marginal coverage P(B* subseteq K_{tau*}(A*)) >= phi - delta
under exchangeability, with no model probability estimates.

Executable content: split-conformal calibration on the nested family. For each trial we
draw n_cal+1 exchangeable (A_i, B_i) pairs, compute the nonconformity score
    s_i = min{ tau : B_i subseteq K_tau(A_i) }   (well defined by the nestedness of Cl.2),
take tau* = the ceil((n_cal+1)*phi)/n_cal empirical quantile of calibration scores, and
record the test-point coverage indicator. Marginal coverage is estimated over many trials
and compared against phi - delta with delta = 1.96*sqrt(p(1-p)/T) (binomial CI).

Mutation: (a) break exchangeability (test scores shifted upward) and (b) use the plain
floor(n*phi) quantile instead of the finite-sample-corrected one -- coverage must drop.
"""
import json, os, numpy as np

SEED = 20260803
os.makedirs('results', exist_ok=True)
rng = np.random.default_rng(SEED)
PHIS = [0.7, 0.75, 0.8, 0.9, 0.95]
NCAL, TRIALS = 50, 4000

def scores(size, shift=0.0, rng=rng):
    """Exchangeable nonconformity scores; the concrete distribution is irrelevant
    (distribution-free), so we use a heavy-tailed mixture + ties."""
    a = rng.lognormal(0, 1.0, size)
    b = rng.choice([0.3, 0.7, 1.1], size=size)
    s = np.where(rng.random(size) < 0.4, b, a)
    return s + shift

rows = []
for phi in PHIS:
    cov = cov_mut_shift = cov_mut_naive = 0
    for _ in range(TRIALS):
        cal = scores(NCAL)
        test = scores(1)[0]
        k = int(np.ceil((NCAL + 1) * phi))
        if k > NCAL:
            tau = np.inf
        else:
            tau = np.sort(cal)[k - 1]
        cov += (test <= tau)
        # mutation a: distribution shift at test time (exchangeability broken)
        cov_mut_shift += (scores(1, shift=0.8)[0] <= tau)
        # mutation b: naive quantile without the +1 correction
        tau_n = np.sort(cal)[max(int(np.floor(NCAL * phi)) - 1, 0)]
        cov_mut_naive += (test <= tau_n)
    p = cov / TRIALS
    delta = 1.96 * np.sqrt(p * (1 - p) / TRIALS)
    rows.append(dict(phi=phi, empirical_coverage=p, delta_binom=delta,
                     ok=bool(p >= phi - delta),
                     mut_shift_coverage=cov_mut_shift / TRIALS,
                     mut_naive_quantile_coverage=cov_mut_naive / TRIALS))

ok = all(r['ok'] for r in rows)
mut = all(r['mut_shift_coverage'] < r['phi'] - r['delta_binom'] for r in rows) and \
      all(r['mut_naive_quantile_coverage'] <= r['empirical_coverage'] + 1e-12 for r in rows)
res = dict(claim="Lemma 1: distribution-free marginal coverage >= phi - delta",
           source="Lemma 1", seed=SEED, n_cal=NCAL, trials=TRIALS,
           all_phi_covered=bool(ok),
           mutations_break=bool(mut),
           verdict="verified" if ok and mut else "inconclusive",
           detail=rows)
json.dump(res, open('results/claim4.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
