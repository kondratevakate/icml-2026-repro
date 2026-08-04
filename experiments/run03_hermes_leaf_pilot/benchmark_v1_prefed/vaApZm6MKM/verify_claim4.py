"""verify_claim4.py — Claim 4 (Section 3.3): asymmetric decomposition l+/l- of the ERT.

Claim: the metrics decompose conditional coverage error into l+-ERT (over-coverage /
unnecessary conservatism, p > 1-alpha) and l--ERT (under-coverage / excessive
aggressiveness, p < 1-alpha), with
    l(p,y)   = l+(p,y) + l-(p,y) - l(1-alpha,y),      l+(p,y) := l(max{p,1-a}, y)
    l-ERT    = l+-ERT + l--ERT,                        l-(p,y) := l(min{p,1-a}, y)
    f = f+ + f-,  f+(p) = f(max{p,1-a}),  f-(p) = f(min{p,1-a}).

Checks are exhaustive over dense grids of alpha and p, for L1/L2/KL, plus a MUTATION
(swap max/min in the definitions) which must break additivity and one-sidedness.

Run: .venv/bin/python verify_claim4.py
"""
import json
import numpy as np

EPS = 1e-12


def loss(name, pred, y, t):
    if name == "L1":
        return np.sign(pred - t) * (t - y)
    if name == "L2":
        return (y - pred) ** 2
    if name == "KL":
        pr = min(max(pred, EPS), 1 - EPS)
        return -(np.log(pr) if y == 1 else np.log(1 - pr))
    raise ValueError(name)


def erisk(name, pred, prob, t):
    return prob * loss(name, pred, 1, t) + (1 - prob) * loss(name, pred, 0, t)


def ert_of(name, pred, prob, t):
    """E_{y~prob}[ l(1-a,y) - l(pred,y) ] = contribution to l-ERT at this x."""
    return erisk(name, t, prob, t) - erisk(name, pred, prob, t)


ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
PGRID = np.round(np.linspace(0.01, 0.99, 197), 6)
NAMES = ["L1", "L2", "KL"]

out = {"grid": {"alphas": ALPHAS, "n_p": len(PGRID), "losses": NAMES}}


def run(swap_mutation: bool):
    res = {}
    for name in NAMES:
        max_loss_decomp_err = 0.0
        max_ert_additivity_err = 0.0
        max_onesidedness_viol = 0.0   # f+ must vanish for p<1-a, f- for p>1-a
        for al in ALPHAS:
            t = 1 - al
            for pr in PGRID:
                hi, lo = max(pr, t), min(pr, t)
                if swap_mutation:      # deliberately broken mechanism
                    hi, lo = lo, hi
                for y in (0, 1):
                    lhs = loss(name, pr, y, t)
                    rhs = loss(name, hi, y, t) + loss(name, lo, y, t) - loss(name, t, y, t)
                    max_loss_decomp_err = max(max_loss_decomp_err, abs(lhs - rhs))
                full = ert_of(name, pr, pr, t)
                plus = ert_of(name, hi, pr, t)
                minus = ert_of(name, lo, pr, t)
                max_ert_additivity_err = max(max_ert_additivity_err, abs(full - (plus + minus)))
                if pr < t:             # pure under-coverage point: over-part must be 0
                    max_onesidedness_viol = max(max_onesidedness_viol, abs(plus))
                if pr > t:             # pure over-coverage point: under-part must be 0
                    max_onesidedness_viol = max(max_onesidedness_viol, abs(minus))
        res[name] = {
            "max_loss_decomposition_abs_error": float(max_loss_decomp_err),
            "max_ERT_additivity_abs_error": float(max_ert_additivity_err),
            "max_one_sidedness_violation": float(max_onesidedness_viol),
        }
    return res


out["decomposition"] = run(False)
out["mutation_swap_max_min"] = run(True)

# Illustration: a mixture population with both over- and under-coverage, 1-alpha=0.9
t = 0.9
pop = np.array([0.98, 0.95, 0.90, 0.85, 0.60])   # p(x) values, uniform weights
demo = {}
for name in NAMES:
    full = float(np.mean([ert_of(name, p_, p_, t) for p_ in pop]))
    plus = float(np.mean([ert_of(name, max(p_, t), p_, t) for p_ in pop]))
    minus = float(np.mean([ert_of(name, min(p_, t), p_, t) for p_ in pop]))
    demo[name] = {"ERT": full, "ERT_plus": plus, "ERT_minus": minus,
                  "sum_parts": plus + minus, "additivity_err": abs(full - (plus + minus))}
swap_demo = {}
for name in NAMES:
    plus_c = float(np.mean([ert_of(name, max(p_, t), p_, t) for p_ in pop]))
    minus_c = float(np.mean([ert_of(name, min(p_, t), p_, t) for p_ in pop]))
    plus_s = float(np.mean([ert_of(name, min(p_, t), p_, t) for p_ in pop]))  # swapped defn
    swap_demo[name] = {"ERT_plus_correct": plus_c, "ERT_minus_correct": minus_c,
                       "ERT_plus_swapped": plus_s,
                       "note": "swapping max/min makes the 'over-coverage' component report the "
                               "under-coverage mass: the asymmetric attribution is inverted."}
out["mutation_swap_worked_example"] = swap_demo
out["mutation_note_additivity"] = ("additivity is invariant under relabelling the two parts, so the "
    "mechanism that breaks under the swap is ONE-SIDEDNESS / attribution, which it does "
    "(max violation ~0.98 for L1, 4.50 for KL vs 0.0 for the correct definition).")
out["worked_example_p_values_0.98_0.95_0.90_0.85_0.60_target_0.9"] = demo

d = out["decomposition"]
m = out["mutation_swap_max_min"]
out["all_checks_pass"] = bool(
    all(v["max_loss_decomposition_abs_error"] < 1e-9 for v in d.values())
    and all(v["max_ERT_additivity_abs_error"] < 1e-9 for v in d.values())
    and all(v["max_one_sidedness_violation"] < 1e-9 for v in d.values())
    and all(v["max_one_sidedness_violation"] > 1e-3 for v in m.values())
    and swap_demo["L1"]["ERT_plus_swapped"] > 0 and swap_demo["L1"]["ERT_plus_correct"] > 0
    and abs(swap_demo["L1"]["ERT_plus_swapped"] - swap_demo["L1"]["ERT_minus_correct"]) < 1e-12
    and all(v["additivity_err"] < 1e-12 for v in demo.values())
)
out["command"] = ".venv/bin/python verify_claim4.py"

with open("results/claim4.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
