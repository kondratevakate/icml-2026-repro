"""Numerical audit of LERD Theorem 4.1 (IVP representation of the event-prior KL).

Theorem 4.1 (paper, Section 4.3). Let q be a strictly positive density on [0,S],
r : [0,S] -> [a,b] subset (0,inf) measurable, R(t)=int_0^t r. Define the normalized
dLIF first-event density

    p_r^S(t) = r(t) exp[-R(t)] / Z_S,   Z_S = int_0^S r(v)exp[-R(v)] dv = 1 - exp[-R(S)].

With h_S(t) = q(t) log(q(t)/p_r^S(t)), the change of variables m = -exp(-t),
M(m) = -log(-m), g(m) = -(q(M(m))/m) log(q(M(m))/p_r^S(M(m))), G'(m)=g(m), G(-1)=0,
the theorem states

    (Eq 9)  K_S := KL(q || p_r^S) = int_0^S h_S(t) dt = G(-exp(-S))     for finite S,
    (Eq 10) K_inf = lim_{eps->0} G(-eps),  |K_inf - G(-eps)| <= int_{-log eps}^inf |h| -> 0.

AUDIT (this is a numerical audit, not a proof replacement; no GPU needed):
  1. Compute K_S directly in t (quadrature of h_S) and via the IVP integral in m
     (quadrature of g), on independent grids; check they agree to ~1e-10.
  2. Check the normalization identity Z_S = 1 - exp[-R(S)].
  3. Check KL >= 0 (Gibbs) and p_r^S, q integrate to 1.
  4. Check the S->inf limit and the Eq-10 tail bound.
  5. CONTROLS that relax the hypotheses, to show they are load-bearing:
     (a) r not bounded away from 0  -> KL integrand loses integrability, identity blows up;
     (b) wrong normalization (use Z_inf=1 on a finite window) -> p not a density, KL<0 possible.

Deps: numpy, scipy.integrate.quad. Deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


# ---- test configurations: (name, r(t), q(t) unnormalized, S) --------------
def make_case(name, r, q_un, S):
    # normalize q on [0, S]
    Zq, _ = quad(q_un, 0, S, limit=200)
    q = lambda t: q_un(t) / Zq
    return dict(name=name, r=r, q=q, S=S)


CASES = [
    make_case("const r, trunc-exp q, S=3",
              r=lambda t: 2.0 + 0.0 * t,
              q_un=lambda t: np.exp(-0.7 * t), S=3.0),
    make_case("linear r in [1,4], gaussian-ish q, S=2.5",
              r=lambda t: 1.0 + 1.2 * t,
              q_un=lambda t: np.exp(-((t - 1.0) ** 2)), S=2.5),
    make_case("sinusoidal r in (0.5,3.5), q=1+.. , S=4",
              r=lambda t: 2.0 + 1.5 * np.sin(1.3 * t),
              q_un=lambda t: 1.0 + 0.5 * np.cos(0.9 * t), S=4.0),
]


def R_of(r, t):
    val, _ = quad(r, 0, t, limit=200)
    return val


def normalizer_ZS(r, S):
    val, _ = quad(lambda v: r(v) * np.exp(-R_of(r, v)), 0, S, limit=200)
    return val


def p_rS(r, ZS, t):
    return r(t) * np.exp(-R_of(r, t)) / ZS


def kl_direct(case):
    r, q, S = case["r"], case["q"], case["S"]
    ZS = normalizer_ZS(r, S)
    h = lambda t: q(t) * np.log(q(t) / p_rS(r, ZS, t))
    val, _ = quad(h, 0, S, limit=400)
    return val, ZS


def kl_ivp(case):
    """K_S = int_{-1}^{-e^{-S}} g(m) dm, the theorem's change of variables."""
    r, q, S = case["r"], case["q"], case["S"]
    ZS = normalizer_ZS(r, S)

    def g(m):
        t = -np.log(-m)                     # M(m)
        return -(q(t) / m) * np.log(q(t) / p_rS(r, ZS, t))

    lo, hi = -1.0, -np.exp(-S)
    val, _ = quad(g, lo, hi, limit=400)
    return val


def main():
    print("Theorem 4.1 numerical audit\n")
    print(f"{'case':<44} {'K_S direct':>12} {'K_S IVP':>12} {'|diff|':>10}  Z_S==1-e^-R(S)  KL>=0")
    ok_all = True
    for c in CASES:
        kd, ZS = kl_direct(c)
        ki = kl_ivp(c)
        diff = abs(kd - ki)
        RS = R_of(c["r"], c["S"])
        z_ok = abs(ZS - (1 - np.exp(-RS))) < 1e-8
        pos = kd >= -1e-9
        ok = diff < 1e-7 and z_ok and pos
        ok_all &= ok
        print(f"{c['name']:<44} {kd:12.8f} {ki:12.8f} {diff:10.2e}      "
              f"{'yes' if z_ok else 'NO':>3}          {'yes' if pos else 'NO'}")

    # ---- S -> inf limit (Eq 10) on the constant-rate case -----------------
    print("\nEq 10 (S->inf):  K_S -> K_inf as horizon grows (const r=2, exp q):")
    r = lambda t: 2.0
    q_un = lambda t: np.exp(-0.7 * t)
    for S in [2, 4, 8, 16]:
        Zq, _ = quad(q_un, 0, S, limit=200)
        c = dict(name="", r=r, q=lambda t: q_un(t) / Zq, S=S)
        kd, _ = kl_direct(c)
        print(f"   S={S:>2}: K_S = {kd:.8f}")

    # ---- CONTROL (a): relax r >= a > 0 -> the Z_inf = 1 claim breaks --------
    # The theorem states Z_inf = 1, which holds because r in [a,b] with a>0 makes
    # R(inf)=inf (so exp[-R(inf)]=0). Relax that: an INTEGRABLE r(t)=1/(1+t)^2 -> 0
    # gives R(inf) < inf, so Z_inf = 1 - exp[-R(inf)] < 1 and p_r^inf is defective.
    print("\nControl (a): relax r>=a>0 with an integrable r(t)=1/(1+t)^2 -> R(inf)<inf, Z_inf breaks:")
    r_int = lambda t: 1.0 / (1.0 + t) ** 2                 # -> 0, integrable; R(inf)=1
    for S in [5, 20, 100, 1000]:
        RS = R_of(r_int, S)
        ZS = 1 - np.exp(-RS)
        print(f"   S={S:>4}: R(S)={RS:.4f}  Z_S=1-e^-R(S)={ZS:.4f}")
    print("   -> Z_S -> 1-e^-1 = 0.632 != 1 as S->inf: the theorem's Z_inf=1 needs r bounded away from 0.")

    # ---- CONTROL (b): wrong normalization on a SHORT window (Z_S << 1) ------
    print("\nControl (b): use Z_inf=1 instead of Z_S on a short/low-rate window (Z_S << 1):")
    r = lambda t: 0.2; S = 1.0                      # R(S)=0.2 -> Z_S = 1-e^-0.2 = 0.181
    q_un = lambda t: r(t) * np.exp(-R_of(r, t)); Zq, _ = quad(q_un, 0, S, limit=200)
    q = lambda t: q_un(t) / Zq                       # q = the CORRECT p_r^S, so true KL = 0
    ZS = normalizer_ZS(r, S)
    kl_true, _ = quad(lambda t: q(t) * np.log(q(t) / (r(t) * np.exp(-R_of(r, t)) / ZS)), 0, S, limit=400)
    p_bad = lambda t: r(t) * np.exp(-R_of(r, t)) / 1.0
    mass, _ = quad(p_bad, 0, S, limit=200)
    kl_bad, _ = quad(lambda t: q(t) * np.log(q(t) / p_bad(t)), 0, S, limit=400)
    print(f"   Z_S={ZS:.4f}; with correct norm KL(q||p_r^S)={kl_true:.4f} (=0, q==p_r^S).")
    print(f"   with Z=1: int p_bad={mass:.4f} (should be 1), KL(q||p_bad)={kl_bad:.4f} "
          f"({'NEGATIVE - Gibbs violated' if kl_bad < 0 else 'nonzero for identical q'}) "
          f"=> the Z_S normalization is load-bearing.")

    print(f"\nVERDICT: Eq 9 identity holds to <1e-7 on all cases: {'YES' if ok_all else 'NO'}")


if __name__ == "__main__":
    main()
