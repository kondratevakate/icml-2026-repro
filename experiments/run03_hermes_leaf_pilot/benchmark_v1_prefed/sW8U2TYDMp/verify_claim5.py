"""Claim 5 (Theorem 21 / Corollary 74, Waluigi shattering).

Under a small-change budget ||DeltaL||_P <= eps and first-order objective
  P'(A) - P(A) = <DeltaL, g_A>_P + o(...),   g_A = 1_A - P(A)   (Lemma 20),
the maximal first-order reduction achievable inside a subspace S is
  M(S) = max_{DeltaL in S, ||DeltaL||_P <= eps} (-<DeltaL, g_A>_P)
       = eps * ||Proj_S g_A||_P    (Lemma 71).
With S0 the baseline span and S1 = S0 + span(w), u = w - Proj_{S0} w:
  M(S1) - M(S0) = eps( ||Proj_{S1} g_A|| - ||Proj_{S0} g_A|| )
                = eps( sqrt(||Proj_{S0}g_A||^2 + <g_A,u>^2/||u||^2) - ||Proj_{S0}g_A|| ) > 0
iff <g_A, u>_P != 0.
Checks: closed form vs. brute-force constrained maximisation; the strict gap.
MUTATION: (a) w already in S0 (u = 0) and (b) u orthogonal to g_A -> gap is
exactly 0, i.e. strictness needs the <g_A,u> != 0 hypothesis that Cor. 74 states
but the main-text wording of Theorem 21 omits.
"""
import numpy as np
from common import rng, rand_simplex, save

r_ = rng(5)
K, d0, eps = 8, 3, 0.02


def ip(a, b, P):
    return float((P * a * b).sum())


def nrm(a, P):
    return float(np.sqrt(max(ip(a, a, P), 0.0)))


def gram_schmidt(vecs, P):
    out = []
    for v in vecs:
        w = v.copy()
        for u in out:
            w = w - ip(w, u, P) * u
        if nrm(w, P) > 1e-10:
            out.append(w / nrm(w, P))
    return out


def proj(g, basis, P):
    return sum(ip(g, b, P) * b for b in basis) if basis else np.zeros_like(g)


gaps, closed_vs_brute = [], 0.0
for _ in range(2000):
    P = rand_simplex(r_, K, conc=2.0)
    A = r_.random(K) < 0.35
    if A.all() or not A.any():
        continue
    gA = A.astype(float) - float(P[A].sum())
    S0v = [r_.normal(size=K) for _ in range(d0)]
    S0v = [v - (P * v).sum() for v in S0v]          # centered log-profile directions
    B0 = gram_schmidt(S0v, P)
    w = r_.normal(size=K); w = w - (P * w).sum()
    u = w - proj(w, B0, P)
    B1 = gram_schmidt(S0v + [w], P)
    M0, M1 = eps * nrm(proj(gA, B0, P), P), eps * nrm(proj(gA, B1, P), P)
    # brute force check of M1 by random search inside the ball of S1
    best = 0.0
    for _ in range(3000):
        c = r_.normal(size=len(B1))
        dL = sum(ci * bi for ci, bi in zip(c, B1))
        nl = nrm(dL, P)
        if nl == 0:
            continue
        dL *= eps / nl
        best = max(best, -ip(dL, gA, P))
    closed_vs_brute = max(closed_vs_brute, abs(best - M1) / max(M1, 1e-12))
    gaps.append(dict(gap=M1 - M0, corr=ip(gA, u, P)))

pos = [g for g in gaps if abs(g["corr"]) > 1e-8]
strict_ok = all(g["gap"] > 1e-14 for g in pos)

# MUTATION (a) w in S0 ; (b) u orthogonal to gA
mut_gaps_a, mut_gaps_b = [], []
for _ in range(500):
    P = rand_simplex(r_, K, conc=2.0)
    A = r_.random(K) < 0.35
    if A.all() or not A.any():
        continue
    gA = A.astype(float) - float(P[A].sum())
    S0v = [r_.normal(size=K) for _ in range(d0)]
    S0v = [v - (P * v).sum() for v in S0v]
    B0 = gram_schmidt(S0v, P)
    w_in = sum(r_.normal() * b for b in B0)                    # (a) w in S0
    B1a = gram_schmidt(S0v + [w_in], P)
    mut_gaps_a.append(eps * nrm(proj(gA, B1a, P), P) - eps * nrm(proj(gA, B0, P), P))
    w = r_.normal(size=K); w = w - (P * w).sum()
    u = w - proj(w, B0, P)
    gperp = gA - proj(gA, B0, P)
    if nrm(gperp, P) < 1e-9:
        continue
    # (b) rebuild u orthogonal to gA within the complement
    z = r_.normal(size=K); z = z - (P * z).sum()
    z = z - proj(z, B0, P)
    z = z - ip(z, gperp, P) / ip(gperp, gperp, P) * gperp
    if nrm(z, P) < 1e-9:
        continue
    B1b = gram_schmidt(S0v + [z], P)
    mut_gaps_b.append(eps * nrm(proj(gA, B1b, P), P) - eps * nrm(proj(gA, B0, P), P))

save(5, dict(
    claim="Theorem 21: manifest-then-suppress ('Waluigi shattering') gives strictly "
          "greater first-order misalignment suppression than pure Luigi reinforcement",
    source="arXiv:2509.06701v2 Sec.5.1 Theorem 21; App. J Lemma 71, Prop. 72, Cor. 74, Thm 75",
    eps=eps, outcome_space=K, baseline_subspace_dim=d0, trials=len(gaps),
    max_rel_err_closed_form_vs_brute_force=closed_vs_brute,
    min_gap_when_corr_nonzero=float(min(g["gap"] for g in pos)),
    median_gap=float(np.median([g["gap"] for g in gaps])),
    strict_inequality_holds_under_Cor74_hypothesis=bool(strict_ok),
    mutation=dict(
        a_w_inside_baseline_span=dict(max_abs_gap=float(np.max(np.abs(mut_gaps_a)))),
        b_u_orthogonal_to_gA=dict(max_abs_gap=float(np.max(np.abs(mut_gaps_b)))),
        property_breaks=bool(np.max(np.abs(mut_gaps_a)) < 1e-12
                             and np.max(np.abs(mut_gaps_b)) < 1e-10)),
    caveat="Strictness requires <g_A, u>_P != 0 and u != 0 (stated in Cor. 74). "
           "The main-text wording of Theorem 21 omits the <g_A,u> != 0 condition; "
           "under the mutation with u orthogonal to g_A the gap is exactly 0, so the "
           "literal main-text statement is false without that hypothesis.",
    verdict="verified (conditional on the Cor. 74 hypothesis)",
))
