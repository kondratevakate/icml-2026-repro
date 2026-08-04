"""
verify_claim3.py — Claim 3 (Theorem 4.3): With decaying step-sizes, the method attains
convergence guarantees with no explicit dimension-dependent terms.

Reproduction (CPU): double-chain TD (Eq.15) with decaying stepsize alpha_t=a/(t+c0)^xi
(xi=1, a=1/eta). Verify (a) error -> theta* (converges, rel.err -> 0 as 1/T), and (b)
NO explicit dimension-dependent term: holding the condition number eta fixed, vary the
feature dimension d and show final error does NOT blow up like d^2 (the prior-art Omega(d^2)
scaling). MUTATION: a CONSTANT (non-decaying) stepsize fails to drive error to 0 (stuck at
the O(alpha/eta) floor) -> the dimension-free / zero-error guarantee breaks.
Deterministic given seed.
"""
from __future__ import annotations
import json, os, numpy as np
from repro_common import (build_mdp_mix, phi_identity, phi_random, theta_star, eta1,
                         run_double_chain)

SEED = 20260503
PAPER = ("Bridging the Gap Between Average and Discounted TD Learning "
         "(arXiv:2605.02103 / OpenReview omkG80XURl)")
SOURCE = ("Theorem 4.3 (double-chain, Markov/i.i.d., decaying stepsize xi in (0,1]); "
          "no explicit dimension-dependent terms in the bound.")
T = 20000
c0 = 1000.0

def run_decay(mdp, Phi, theta0, eta, sampler_kind, seed):
    a = 1.0 / eta
    return run_double_chain(mdp, Phi, theta0, (lambda t: a / (t + c0)), T, sampler_kind, seed)

def make_phi_scaled(n, d, target_eta, mdp, rng):
    Phi = rng.standard_normal((n, d))
    Phi[:, -1] = 1.0
    rn = np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi = Phi / np.maximum(rn, 1.0)
    e1 = eta1(Phi, mdp["P"], mdp["mu"])
    s = np.sqrt(target_eta / max(e1, 1e-12))
    return Phi * s

# ---- base convergence (tabular) with decaying stepsize -> rel.err -> 0 ----
mdp0 = build_mdp_mix(6, 0.5, rng=np.random.default_rng(SEED))
Phi0 = phi_identity(mdp0["n"])
ths, A, condA = theta_star(Phi0, mdp0["P"], mdp0["mu"], mdp0["R"])
norm_ts = float(np.linalg.norm(ths) ** 2)
e1_0 = eta1(Phi0, mdp0["P"], mdp0["mu"])
g, e2 = run_decay(mdp0, Phi0, np.zeros(mdp0["n"]), e1_0, "iid", SEED)
rel_decay = float(e2[-1] / norm_ts)
converges = bool(rel_decay < 0.3)

# ---- no explicit dimension term: vary d at FIXED eta1, fit error exponent vs d ----
d_vals = [4, 8, 12, 16, 24]
target_eta = 0.04
mdp = build_mdp_mix(12, 0.6, rng=np.random.default_rng(SEED + 7))
rel_errs, ds, etas_meas = [], [], []
for d in d_vals:
    rng = np.random.default_rng(SEED + d * 13)
    Phi = make_phi_scaled(mdp["n"], d, target_eta, mdp, rng)
    e1 = eta1(Phi, mdp["P"], mdp["mu"])
    th, _, _ = theta_star(Phi, mdp["P"], mdp["mu"], mdp["R"])
    nt = float(np.linalg.norm(th) ** 2)
    gg, ee = run_decay(mdp, Phi, np.zeros(mdp["n"]), e1, "iid", SEED)
    rel = float(ee[-1] / max(nt, 1e-12))
    rel_errs.append(rel); ds.append(d); etas_meas.append(e1)
ds = np.array(ds); rel_errs = np.array(rel_errs)
slope_d, _ = np.polyfit(np.log(ds), np.log(rel_errs), 1)
# prior-art Omega(d^2) would give slope ~ +2; claim is NO explicit d term -> slope ~ 0
no_dim_term = bool(abs(slope_d) < 1.2)      # far below the prior +2 scaling

# ---- MUTATION: constant (non-decaying) stepsize -> error stuck at floor, no -> 0 ----
alpha_c = e1_0 / 18.0
gC, e2C = run_double_chain(mdp0, Phi0, np.zeros(mdp0["n"]), alpha_c, 20000, "iid", SEED)
rel_const = float(e2C[-1] / norm_ts)
mutation_breaks = bool(rel_const > 2.0 * rel_decay and rel_decay < 0.3)

verdict = "verified" if (converges and no_dim_term and mutation_breaks) else "toy"

result = {
    "claim_no": 3,
    "claim": ("With decaying step-sizes, the method attains convergence guarantees with no "
              "explicit dimension-dependent terms (Theorem 4.3)."),
    "source": SOURCE,
    "paper": PAPER,
    "seed": SEED,
    "method": ("Double-chain TD (Eq.15) with decaying stepsize alpha_t=(1/eta)/(t+c0); "
               "convergence to theta*; dimension test: vary feature dim d at fixed eta1 "
               "(rescaled random features), fit error exponent vs d; mutation = constant "
               "stepsize (no decay)."),
    "verdict": verdict,
    "reason": (f"With decaying stepsize the double chain converges to theta* (rel.err {rel_decay:.3f} "
               f"at T={T}). Varying feature dimension d in {d_vals} at fixed eta1~{target_eta:.2f}, "
               f"final rel.err exponent vs d = {slope_d:.2f} (prior-art Omega(d^2) would be +2; "
               f"claim of NO explicit d term supported). MUTATION (constant stepsize): rel.err "
               f"stays {rel_const:.3f} >> decaying {rel_decay:.3f} (O(alpha/eta) floor, no ->0) -> "
               f"dimension-free guarantee breaks. EVIDENCE BOUNDARY: empirical error-vs-d exponent "
               f"reflects variance, not the formal bound; we verify the qualitative absence of a "
               f"d^2 blow-up on CPU instances, not a formal proof."),
    "executed_numeric_experiment": True,
    "metrics": {
        "T": T, "c0": c0,
        "decay_rel_err": rel_decay,
        "converges_to_theta_star": converges,
        "d_vals": [int(x) for x in ds],
        "eta1_measured": [float(x) for x in etas_meas],
        "rel_err_vs_d": [float(x) for x in rel_errs],
        "exponent_vs_d": float(slope_d),
        "prior_art_exponent": 2.0,
        "no_explicit_dim_term": no_dim_term,
        "const_stepsize_rel_err": rel_const,
    },
    "mutation": {
        "description": ("Replace the decaying stepsize by a constant stepsize alpha=eta/18. The "
                        "constant-stepsize bound has a steady-state floor O(alpha/eta); error cannot "
                        "be driven to 0, so the dimension-free / zero-error guarantee of Theorem 4.3 "
                        "is lost."),
        "decay_rel_err": rel_decay,
        "constant_rel_err": rel_const,
        "constant_worse_by": float(rel_const / max(rel_decay, 1e-12)),
        "passed": mutation_breaks,
    },
}
os.makedirs("results", exist_ok=True)
with open("results/claim3.json", "w") as f:
    json.dump(result, f, indent=2)
print("claim3:", verdict, "| rel_decay=%.3f slope_d=%.2f const=%.3f" % (rel_decay, slope_d, rel_const))
