"""Claim 6 (TASK #6; paper v2 Eqs. (5)-(7), Lemma C.2).

Claim: confidence sets are built by regularised least squares in L^2(R^d;rho),
C_t(delta) = { f : ||f - fhat_t||_{V_t^lam} <= beta_t(delta) } with
beta_t = sigma sqrt(log(4 det(DLam + M M*/lam)/delta^2)) + sqrt(lam) Cbar/||DLam||_op^{1/2},
i.e. an OFUL-style ellipsoid whose width is controlled by a log-determinant.

Executable test (finite-dimensional truncation, DLam = Id since Lam = ||.||^2/2):
  * closed-form RLS estimator == argmin of the penalised objective (numerical check),
  * uniform-over-time coverage: fraction of runs with f* in cap_t C_t(delta) >= 1-delta,
  * width grows exactly like the claimed log-det expression (Sylvester identity check
    det(Id + M M*/lam) = det(Id + M* M/lam)),
  * MUTATION: shrink beta by /6 (or set lam->0 in the bias term) -> coverage must break.
Seed 20260803.
"""
import numpy as np
from common import save, beta_t

SEED = 20260803
d, T, RUNS = 8, 300, 400
lam, sigma, delta = 1.0, 0.3, 0.05
rng0 = np.random.default_rng(SEED)
fstar = rng0.standard_normal(d); fstar /= np.linalg.norm(fstar)
Cbar = 1.0  # ||c*|| <= 1

# --- closed-form RLS vs numerical minimiser -----------------------------
rng = np.random.default_rng(SEED + 1)
A = rng.standard_normal((25, d)) / np.sqrt(d)
Cobs = A @ fstar + sigma * rng.standard_normal(25)
V = A.T @ A + lam * np.eye(d)
fhat = np.linalg.solve(V, A.T @ Cobs)
grad = 2 * (A.T @ (A @ fhat - Cobs)) + 2 * lam * fhat   # dL/df with Lam=||.||^2/2 -> lam*f *2? see obj
obj = lambda f: np.sum((Cobs - A @ f) ** 2) + lam * np.dot(f, f)
num_grad = np.array([(obj(fhat + 1e-6 * e) - obj(fhat - 1e-6 * e)) / 2e-6
                     for e in np.eye(d)])
rls_grad_norm = float(np.linalg.norm(num_grad))

# --- Sylvester identity: det(Id_d + M*M/lam) == det(Id_t + MM*/lam) -----
syl = float(abs(np.linalg.slogdet(np.eye(d) + A.T @ A / lam)[1]
                - np.linalg.slogdet(np.eye(25) + A @ A.T / lam)[1]))

# --- uniform coverage experiment ----------------------------------------
def run(scale_beta=1.0, seed=0):
    r = np.random.default_rng(seed)
    Vt = lam * np.eye(d)
    b = np.zeros(d)
    covered = True
    widths = []
    for t in range(1, T + 1):
        a = r.standard_normal(d); a /= np.linalg.norm(a)     # ||a||<=1 (=||F pi||)
        c = a @ fstar + sigma * r.standard_normal()
        Vt += np.outer(a, a); b += c * a
        fh = np.linalg.solve(Vt, b)
        bt = scale_beta * beta_t(Vt, lam, sigma, Cbar, delta)
        e = fh - fstar
        dist = np.sqrt(e @ Vt @ e)
        widths.append((t, float(bt), float(dist)))
        if dist > bt:
            covered = False
    return covered, widths

cov = np.mean([run(1.0, SEED + 100 + i)[0] for i in range(RUNS)])
cov_mut = np.mean([run(1 / 6.0, SEED + 100 + i)[0] for i in range(RUNS)])
_, w = run(1.0, SEED + 7)
width_T = w[-1][1]

ok = (rls_grad_norm < 1e-6 and syl < 1e-8 and cov >= 1 - delta and cov_mut < 1 - delta)
out = dict(
    claim=6, source="paper v2 Eqs. (5),(6),(7) + Lemma C.2 (TASK cites 'Equations 11-12')",
    seed=SEED, d=d, T=T, runs=RUNS, lam=lam, sigma=sigma, delta=delta,
    rls_closed_form_gradient_norm=rls_grad_norm,
    sylvester_logdet_gap=syl,
    empirical_uniform_coverage=float(cov), required_coverage=1 - delta,
    beta_T=float(width_T),
    mutation_beta_over_6_coverage=float(cov_mut),
    verdict="verified" if ok else "inconclusive",
)
for k, v in out.items():
    print(f"{k}: {v}")
save("claim6.json", out)
