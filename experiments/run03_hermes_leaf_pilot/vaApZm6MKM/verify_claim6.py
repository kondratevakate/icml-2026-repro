"""verify_claim6.py — Claim 6 (Algorithm 1).

Claim: Algorithm 1 estimates the ERT metrics from finite samples using k-fold cross-validation
to avoid overfitting the classifier used in the estimation.

Test design (simulation, exhaustive over 20 seeds x 3 sample sizes x 2 regimes):
  Regime "perfect": Z ~ Bernoulli(1-alpha) INDEPENDENT of X (perfect conditional coverage).
      Ground truth ERT = 0. Algorithm 1 (k-fold) must return ~0 (no false detection).
  Regime "violated": p(x) = 0.9 + 0.25*(x1-0.5)*2 (genuine conditional miscoverage) with known
      analytic ground truth E|p(X)-(1-alpha)|; Algorithm 1 must be positive and (being a lower
      bound) not exceed the true value by more than noise.
  MUTATION (the mechanism broken deliberately): cross_fit=False, i.e. the same classifier is
      trained and evaluated on all points, exactly what Algorithm 1's k-fold step prevents.
      Prediction: in the "perfect" regime the no-CV estimate becomes strongly POSITIVE
      (spurious detection of conditional-coverage failure) while the k-fold estimate stays ~0.

Run: .venv/bin/python verify_claim6.py
"""
import json
import numpy as np
from ertlib import algorithm1

ALPHA = 0.1
T = 1 - ALPHA
SEEDS = list(range(20))
SIZES = [500, 1000, 2000]
D = 5


def gen(regime, n, rng):
    X = rng.uniform(0, 1, size=(n, D))
    if regime == "perfect":
        p = np.full(n, T)
    else:
        p = np.clip(T + 0.5 * (X[:, 0] - 0.5), 0.0, 1.0)
    Z = (rng.uniform(size=n) < p).astype(float)
    return X, Z, p


# analytic ground truth for the "violated" regime: E|p(X)-0.9| with p = 0.9+0.5(U-0.5)
TRUE_VIOLATED = {"L1": 0.125, "L2": float(np.mean((0.5 * (np.linspace(0, 1, 2_000_001) - .5)) ** 2))}

records = []
for regime in ["perfect", "violated"]:
    for n in SIZES:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * s + n)
            X, Z, p = gen(regime, n, rng)
            cv = algorithm1(X, Z, T, losses=("L1", "L2"), k=5, clf_kind="gb", seed=s, cross_fit=True)
            nocv = algorithm1(X, Z, T, losses=("L1", "L2"), k=5, clf_kind="gb", seed=s, cross_fit=False)
            print(regime, n, s, flush=True)
            records.append({"regime": regime, "n": n, "seed": s,
                            "cv_L1": cv["L1"], "cv_L2": cv["L2"],
                            "nocv_L1": nocv["L1"], "nocv_L2": nocv["L2"],
                            "empirical_true_L1": float(np.mean(np.abs(p - T)))})

summary = {}
for regime in ["perfect", "violated"]:
    for n in SIZES:
        r = [x for x in records if x["regime"] == regime and x["n"] == n]
        summary[f"{regime}_n{n}"] = {
            "cv_L1_mean": float(np.mean([x["cv_L1"] for x in r])),
            "cv_L1_std": float(np.std([x["cv_L1"] for x in r])),
            "nocv_L1_mean": float(np.mean([x["nocv_L1"] for x in r])),
            "nocv_L1_std": float(np.std([x["nocv_L1"] for x in r])),
            "cv_L2_mean": float(np.mean([x["cv_L2"] for x in r])),
            "nocv_L2_mean": float(np.mean([x["nocv_L2"] for x in r])),
            "true_L1": float(np.mean([x["empirical_true_L1"] for x in r])),
            "n_seeds": len(r),
            "frac_seeds_cv_L1_positive": float(np.mean([x["cv_L1"] > 0.01 for x in r])),
            "frac_seeds_nocv_L1_positive": float(np.mean([x["nocv_L1"] > 0.01 for x in r])),
        }

checks = {
    # k-fold does not hallucinate miscoverage under perfect conditional coverage
    "cv_near_zero_in_perfect_regime": all(
        abs(summary[f"perfect_n{n}"]["cv_L1_mean"]) < 0.02 for n in SIZES),
    # the broken (no-CV) variant does hallucinate it
    "nocv_strongly_positive_in_perfect_regime": all(
        summary[f"perfect_n{n}"]["nocv_L1_mean"] > 0.05 for n in SIZES),
    "nocv_beats_cv_by_factor>3_in_perfect_regime": all(
        summary[f"perfect_n{n}"]["nocv_L1_mean"] >
        3 * max(abs(summary[f"perfect_n{n}"]["cv_L1_mean"]), 1e-6) for n in SIZES),
    # in a genuinely violated regime the CV estimate is positive and a lower bound on the truth
    "cv_positive_in_violated_regime": all(
        summary[f"violated_n{n}"]["cv_L1_mean"] > 0.02 for n in SIZES),
    "cv_is_lower_bound_in_violated_regime": all(
        summary[f"violated_n{n}"]["cv_L1_mean"] <= summary[f"violated_n{n}"]["true_L1"] + 0.01
        for n in SIZES),
}

out = {"alpha": ALPHA, "d": D, "seeds": SEEDS, "sizes": SIZES,
       "true_L1_violated_analytic": TRUE_VIOLATED["L1"],
       "summary": summary, "checks": checks,
       "all_checks_pass": bool(all(checks.values())),
       "records": records,
       "command": ".venv/bin/python verify_claim6.py"}
with open("results/claim6.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))
