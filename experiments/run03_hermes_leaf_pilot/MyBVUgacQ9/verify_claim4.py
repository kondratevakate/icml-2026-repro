"""verify_claim4.py -- CLAIM 4 / Corollary 4.2 (Section 4).

Claim: for the locomotion problem (eq. 5 -> eq. 6) linear convergence holds when the
discretisation step Dt is below a threshold t0, "since the nonlinear terms decay as
O((Dt)^3)".

Two independent parts:

  (A) SYMBOLIC (sympy).  Derive eq. 6 from eq. 5 for a 2-D locomotion instance
      (c in R^2, angular momentum k scalar, N contacts): eliminate c and c_dot,
      form k'_{i+1} = k_{i+1} - k_i, expand, and split into terms of degree 1 and
      degree 2 in the forces f.  Check: every degree-2 (multi-affine quadratic)
      coefficient is exactly proportional to Dt^3, and check that the C_i matrices
      built by our numeric model reproduce the sympy expansion coefficient by
      coefficient.  Then ||C(Dt)|| is measured on a Dt grid and the exponent p in
      ||C|| ~ Dt^p is fitted (expected p = 3).

  (B) SIMULATION.  Run Algorithm 1 on the eq.-6 problem (m = 2 kg as in Fig. 5,
      f(f) = 0.5 sum ||f_i||^2 + I_i(f_i), phi(z) = 5 sum ||k'_i||^2, polyhedral
      (box) force constraints) for a grid of Dt and several random initialisations,
      and measure the asymptotic per-iteration contraction factor.  Corollary 4.2
      predicts: linear (factor bounded away from 1) for small Dt.

  MUTATION.  Freeze the nonlinear coefficient so that it does NOT scale with Dt
  (multiply the quadratic block by Dt^-3, keeping the linear part untouched):
  the Dt-threshold mechanism must disappear / the rate must degrade.

Run:  .venv/bin/python verify_claim4.py
"""
import json
import time

import numpy as np
import sympy as sp

import admm_lib as A

OUT = "results/claim4.json"
CMD = ".venv/bin/python verify_claim4.py"

M_ROBOT = 2.0          # kg, Fig. 5 (Solo, Grimminger et al. 2020)
GRAV = np.array([0.0, -9.81])
N_CONTACT = 2


def contact_positions(T, N=N_CONTACT):
    """fixed contact locations r_i^j (known, eq. 5)."""
    r = np.zeros((T, N, 2))
    for i in range(T):
        for j in range(N):
            r[i, j] = [0.05 * i + 0.15 * j - 0.1, 0.0]
    return r


def build_eq6(dt, T, c_init=np.array([0.0, 0.3]), cd_init=np.array([0.1, 0.0]),
              k_init=0.05, m=M_ROBOT, quad_scale=1.0, box=True):
    """Multi-affine quadratic problem of eq. 6 for the planar case.
    x = [f_0^1, f_0^2, ..., f_{T-1}^1, f_{T-1}^2] in R^{4T} (blocks = one time step),
    z = [k'_0, ..., k'_T] in R^{T+1}, Q = I  (full row rank -> Assumption 2.6 holds).
    quad_scale != 1 rescales ONLY the quadratic (multi-affine) part -> mutation knob.
    """
    N = N_CONTACT
    r = contact_positions(T, N)
    nx = 4 * T
    nc = T + 1
    nz = T + 1

    def idx(i, j, comp):
        return 4 * i + 2 * j + comp

    C = np.zeros((nc, nx, nx))
    d = np.zeros((nc, nx))
    e = np.zeros(nc)
    e[0] = -k_init                                   # row 0: k'_0 - k_init = 0

    for i in range(T):
        row = i + 1
        B = np.zeros((nx, nx))
        # linear part: P_i^j = r_i^j - c_init - cd_init*i*dt - sum_{i'<=i-2}(i-1-i')*g*dt^2
        gsum = np.zeros(2)
        for ip in range(0, max(i - 1, 0)):
            gsum = gsum + (i - 1 - ip) * GRAV * dt ** 2
        for j in range(N):
            P = r[i, j] - c_init - cd_init * i * dt - gsum
            # -dt * (P x f) = -dt*(P0 f1 - P1 f0)
            d[row, idx(i, j, 0)] += dt * P[1]
            d[row, idx(i, j, 1)] += -dt * P[0]
            # quadratic part: +dt^3/m * sum_{i'<=i-2} (i-1-i') * (F_{i'} x f_i^j)
            for ip in range(0, max(i - 1, 0)):
                kap = quad_scale * dt ** 3 * (i - 1 - ip) / m
                for l in range(N):
                    B[idx(ip, l, 0), idx(i, j, 1)] += kap
                    B[idx(ip, l, 1), idx(i, j, 0)] += -kap
        C[row] = B + B.T

    Q = np.eye(nc, nz)
    mu_x = np.ones(nx)          # f(f) = 0.5 sum ||f_i||^2
    mu_z = 10.0 * np.ones(nz)   # phi(z) = 5 sum ||k'_i||^2
    bx = None
    if box:
        fx = 0.6 * m * 9.81
        fz_lo, fz_hi = 0.3 * m * 9.81, 2.0 * m * 9.81
        lo = np.array([-fx, fz_lo] * N)
        hi = np.array([fx, fz_hi] * N)
        bx = [(lo, hi)] * T
    blocks = [list(range(4 * i, 4 * i + 4)) for i in range(T)]
    return A.MultiAffineProblem(blocks, C, d, e, Q, mu_x, mu_z, box=bx)


# ---------------------------------------------------------------- part A
def symbolic_check(T=4, N=N_CONTACT):
    dt = sp.Symbol("dt", positive=True)
    m = sp.Rational(2)
    gx, gy = sp.Integer(0), sp.Rational(-981, 100)
    c0 = sp.Matrix([0, sp.Rational(3, 10)])
    cd0 = sp.Matrix([sp.Rational(1, 10), 0])
    f = [[sp.Matrix([sp.Symbol("f_%d_%d_x" % (i, j)), sp.Symbol("f_%d_%d_z" % (i, j))])
          for j in range(N)] for i in range(T)]
    rnum = contact_positions(T, N)
    r = [[sp.Matrix([sp.nsimplify(rnum[i, j, 0]), sp.nsimplify(rnum[i, j, 1])])
          for j in range(N)] for i in range(T)]

    # eq. 5 forward recursion
    c = [c0]
    cd = [cd0]
    for i in range(T):
        c.append(c[i] + cd[i] * dt)
        F = sp.Matrix([0, 0])
        for j in range(N):
            F += f[i][j]
        cd.append(cd[i] + F / m * dt + sp.Matrix([gx, gy]) * dt)

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    report = []
    for i in range(T):
        kp = sp.expand(sum(cross(r[i][j] - c[i], f[i][j]) for j in range(N)) * dt)
        poly = sp.Poly(kp, *[s for row in f for v in row for s in v])
        deg2, deg1 = sp.Integer(0), sp.Integer(0)
        for mono, coef in poly.terms():
            term = coef * sp.prod([s ** p for s, p in
                                   zip(poly.gens, mono)])
            if sum(mono) == 2:
                deg2 += term
            elif sum(mono) == 1:
                deg1 += term
        # every quadratic coefficient must be divisible by dt^3
        quad_ok = True
        quad_exponents = []
        for mono, coef in sp.Poly(sp.expand(deg2), *poly.gens).terms():
            cc = sp.simplify(coef)
            if cc == 0:
                continue
            pw = sp.degree(sp.Poly(cc, dt), dt)
            quad_exponents.append(int(pw))
            if pw != 3:
                quad_ok = False
        lin_exponents = sorted({int(sp.degree(sp.Poly(sp.simplify(coef), dt), dt))
                                for mono, coef in sp.Poly(sp.expand(deg1), *poly.gens).terms()
                                if sp.simplify(coef) != 0})
        report.append({"row": i + 1, "n_quadratic_terms": len(quad_exponents),
                       "quadratic_dt_exponents": sorted(set(quad_exponents)),
                       "all_quadratic_dt_pow3": quad_ok,
                       "linear_dt_exponents": lin_exponents})
    return report


def symbolic_vs_numeric(T=4, dt_val=0.02, N=N_CONTACT):
    """Check the numeric C_i, d_i built by build_eq6 reproduce the sympy expansion."""
    dt = sp.Symbol("dt", positive=True)
    prob = build_eq6(dt_val, T)
    m = sp.Rational(2)
    c0 = sp.Matrix([0, sp.Rational(3, 10)])
    cd0 = sp.Matrix([sp.Rational(1, 10), 0])
    fs = [[sp.Matrix([sp.Symbol("f_%d_%d_x" % (i, j)), sp.Symbol("f_%d_%d_z" % (i, j))])
           for j in range(N)] for i in range(T)]
    gens = [s for row in fs for v in row for s in v]
    rnum = contact_positions(T, N)
    c = [c0]; cd = [cd0]
    for i in range(T):
        c.append(c[i] + cd[i] * dt)
        F = sp.Matrix([0, 0])
        for j in range(N):
            F += fs[i][j]
        cd.append(cd[i] + F / m * dt + sp.Matrix([0, sp.Rational(-981, 100)]) * dt)

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    maxerr = 0.0
    rng = np.random.default_rng(0)
    for _ in range(5):
        xv = rng.normal(size=4 * T) * 5
        subs = {g: float(xv[i]) for i, g in enumerate(gens)}
        subs[dt] = dt_val
        Anum = prob.A(xv)
        for i in range(T):
            kp = sum(cross(sp.Matrix([sp.nsimplify(rnum[i, j, 0]), sp.nsimplify(rnum[i, j, 1])])
                           - c[i], fs[i][j]) for j in range(N)) * dt
            sym_val = float(sp.expand(kp).subs(subs))
            # our row is (A)_{i+1} = -(that expression)
            maxerr = max(maxerr, abs(Anum[i + 1] + sym_val))
    return float(maxerr)


def cnorm_scaling(T=6, dts=(0.002, 0.005, 0.01, 0.02, 0.05)):
    xs, ys = [], []
    for dt in dts:
        p = build_eq6(dt, T)
        xs.append(np.log(dt)); ys.append(np.log(p.cnorm()))
    slope, intercept = np.polyfit(xs, ys, 1)
    return {"dts": list(dts),
            "C_norms": [float(np.exp(y)) for y in ys],
            "fitted_exponent_p_in_Cnorm_prop_dt_p": float(slope)}


# ---------------------------------------------------------------- part B
def rate_for_dt(dt, T=6, rho=4.0, seeds=(0, 1, 2, 3), iters=140, quad_scale=1.0):
    prob = build_eq6(dt, T, quad_scale=quad_scale)
    facs, viols = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        x0 = np.zeros(4 * T)
        for i in range(T):
            for j in range(N_CONTACT):
                x0[4 * i + 2 * j] = rng.uniform(-5, 5)
                x0[4 * i + 2 * j + 1] = rng.uniform(0.3 * M_ROBOT * 9.81,
                                                    2 * M_ROBOT * 9.81)
        z0 = rng.normal(size=T + 1) * 0.05
        x, z, w, h = prob.run(x0, z0, np.zeros(T + 1), rho, iters=iters)
        if not np.all(np.isfinite(np.concatenate([x, z, w]))):
            facs.append(float("inf")); viols.append(float("inf")); continue
        f, npts, r2 = A.contraction_factor(h["res"], floor=1e-13)
        if not np.isfinite(f):
            rr = h["res"]; idx = np.where(rr > 1e-13)[0]
            f = float(np.exp((np.log(rr[idx[-1]]) - np.log(rr[idx[0]])) /
                             max(1, idx[-1] - idx[0]))) if len(idx) >= 3 else 0.0
        facs.append(float(f)); viols.append(float(h["viol"][-1]))
    return {"dt": dt, "C_norm": prob.cnorm(), "worst_factor": float(np.max(facs)),
            "median_factor": float(np.median(facs)),
            "worst_final_viol": float(np.max(viols)), "factors": facs}


def main():
    t0 = time.time()
    res = {"claim": 4, "source": "Corollary 4.2 (+ eq. 6), Section 4", "command": CMD}

    res["A_symbolic_dt_exponents"] = symbolic_check(T=4)
    res["A_symbolic_vs_numeric_max_abs_err"] = symbolic_vs_numeric(T=4)
    res["A_Cnorm_scaling"] = cnorm_scaling()

    dts = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    res["B_rate_vs_dt"] = [rate_for_dt(dt) for dt in dts]
    res["B_MUTATION_dt_independent_nonlinearity"] = [
        rate_for_dt(dt, quad_scale=1.0 / dt ** 3) for dt in dts]

    res["elapsed_sec"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    print("A) symbolic expansion of eq. 6 (T=4):")
    for r in res["A_symbolic_dt_exponents"]:
        print("   row %d: %d quadratic terms, dt exponents %s (all = 3: %s); "
              "linear-term dt exponents %s" %
              (r["row"], r["n_quadratic_terms"], r["quadratic_dt_exponents"],
               r["all_quadratic_dt_pow3"], r["linear_dt_exponents"]))
    print("   symbolic vs numeric model max abs error: %.3e"
          % res["A_symbolic_vs_numeric_max_abs_err"])
    print("   fitted exponent of ||C|| ~ dt^p : %.4f"
          % res["A_Cnorm_scaling"]["fitted_exponent_p_in_Cnorm_prop_dt_p"])
    print("B) ADMM on eq. 6 (T=6, m=2kg, rho=4, 4 initialisations):")
    for r in res["B_rate_vs_dt"]:
        print("   dt=%6.3f  ||C||=%9.3e  worst factor %.4f  median %.4f  viol %.1e"
              % (r["dt"], r["C_norm"], r["worst_factor"], r["median_factor"],
                 r["worst_final_viol"]))
    print("   MUTATION (nonlinearity forced dt-independent, ||C|| fixed):")
    for r in res["B_MUTATION_dt_independent_nonlinearity"]:
        print("   dt=%6.3f  ||C||=%9.3e  worst factor %.4f  viol %.1e"
              % (r["dt"], r["C_norm"], r["worst_factor"], r["worst_final_viol"]))
    print("wrote", OUT, "in %.1fs" % res["elapsed_sec"])


if __name__ == "__main__":
    main()
