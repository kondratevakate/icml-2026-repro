"""
Independent reproduction of ConfSleepNet (ICML 2026), Claim 5:
Proposition 1 and Proposition 2 about uncertainty under conflict-aware aggregation.

No code was released by the authors (repo has only a README), so this is an
independent reimplementation of the paper's equations, transcribed from the PDF:

  Eq 4  : b_k^v = (alpha_k^v - 1)/S^v ,  u^v = K/S^v ,  K=5
  Eq 5  : sum_k b_k^v + u^v = 1
  Eq 6  : C(Ma,Mb) = 1 - (sum_k b_k^a b_k^b) / ((sum_i b_i^a)(sum_j b_j^b))
  Eq 8  : u^{a|b} = (1-C) u_a u_b + C^2 u_a u_b / (u_a + u_b)
  Eq 9  : b_k^{a|b} = [u_a b_k^b + u_b b_k^a + (1-C) u_a u_b (b_k^a + b_k^b)] / (u_a + u_b)

We construct two opinions and sweep their conflict C from 0 to 1, then check the
two propositions. We compute the combined uncertainty two ways:
  u_eq8  : Eq 8 exactly as printed
  u_norm : 1 - sum_k b_k^{a|b}, i.e. forced consistent with the Eq 5 constraint
"""
import numpy as np

K = 5


def conflict(ba, bb):                      # Eq 6
    num = float(np.sum(ba * bb))
    den = float(np.sum(ba) * np.sum(bb))
    return 1.0 - num / den


def combine_belief(ba, ua, bb, ub, C):     # Eq 9
    return (ua * bb + ub * ba + (1.0 - C) * ua * ub * (ba + bb)) / (ua + ub)


def u_eq8(ua, ub, C):                       # Eq 8 as printed
    return (1.0 - C) * ua * ub + C ** 2 * ua * ub / (ua + ub)


# base opinion Mo (uncertainty uo), all belief on class 0
uo = 0.3
bo = np.zeros(K); bo[0] = 1.0 - uo

# second opinion Mx with HIGHER uncertainty (ub > uo), required by Proposition 2
ux = 0.6
bx_mass = 1.0 - ux

print("t      C       u_norm   u_eq8    sum_b+u_norm")
rows = []
for t in np.linspace(0.0, 1.0, 11):
    bx = np.zeros(K)
    bx[0] = (1.0 - t) * bx_mass            # aligned part (class 0)
    bx[1] = t * bx_mass                     # opposed part (class 1)
    C = conflict(bo, bx)
    b_comb = combine_belief(bo, uo, bx, ux, C)
    un = 1.0 - float(b_comb.sum())          # normalization-consistent u
    u8 = u_eq8(uo, ux, C)
    rows.append((t, C, un, u8))
    print(f"{t:4.2f}  {C:6.4f}  {un:7.4f}  {u8:7.4f}   {b_comb.sum()+un:7.4f}")

C0 = rows[0]     # C ~ 0  (consistent)
C1 = rows[-1]    # C ~ 1  (fully conflicting)

print("\n--- analytic limits ---")
print(f"uo = {uo}, ux = {ux} (ux > uo, as Prop 2 requires)")
print(f"C->0 : u_a*u_b            = {uo*ux:.4f}")
print(f"C->1 : u_a*u_b/(u_a+u_b)  = {uo*ux/(uo+ux):.4f}   (this is Eq 8's limit)")
print(f"C->1 : 2*u_a*u_b/(u_a+u_b)= {2*uo*ux/(uo+ux):.4f}   (this is the 1-sum(b) limit)")

def verdict(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

print("\n--- Proposition checks (normalization-consistent u_norm) ---")
p1_norm = C0[2] < uo
p2_norm = C1[2] > uo
verdict("Prop 1  (C->0): u_norm < uo  (uncertainty decreases)", p1_norm)
verdict("Prop 2  (C->1): u_norm > uo  (uncertainty increases)", p2_norm)

print("\n--- Same checks using Eq 8 exactly as printed ---")
p1_eq8 = C0[3] < uo
p2_eq8 = C1[3] > uo
verdict("Prop 1  (C->0): u_eq8 < uo", p1_eq8)
verdict("Prop 2  (C->1): u_eq8 > uo", p2_eq8)

print("\n--- normalization Eq 5 (sum_k b + u = 1) ---")
sums = [abs(r[2] + (1 - r[2]) - 1) for r in rows]  # trivially true for u_norm by construction
eq8_breaks_norm = abs(C1[3] - C1[2]) > 1e-6
verdict("u_norm satisfies sum(b)+u = 1 by construction", True)
print(f"[NOTE] at C~1: u_eq8={C1[3]:.4f} vs u_norm={C1[2]:.4f} -> "
      f"Eq 8 {'DIFFERS from' if eq8_breaks_norm else 'matches'} the normalization value")

print("\n=== SUMMARY ===")
print("Propositions 1 and 2 REPRODUCE when uncertainty is taken consistent with")
print("the paper's own constraint sum_k b + u = 1 (Eq 5 / Appendix A.5).")
print("Eq 8 as printed reproduces Prop 1 but NOT Prop 2 (its C->1 limit is a")
print("decrease), and it disagrees with the normalization by a factor ~2 at high")
print("conflict -> Eq 8 as printed appears to be missing a normalization factor.")
