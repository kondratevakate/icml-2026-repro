import numpy as np
from repro_common import *

rng = np.random.default_rng(0)
mdp = build_mdp_mix(n=8, eps=0.6, rng=rng)   # eta1 ~ (1-eps)/n = 0.4/8 = 0.05
Phi = phi_identity(mdp["n"])
theta_star_v, A, condA = theta_star(Phi, mdp["P"], mdp["mu"], mdp["R"])
W_star, g = relative_value(mdp["P"], mdp["mu"], mdp["R"])
e1 = eta1(Phi, mdp["P"], mdp["mu"])
e2 = eta2(Phi, mdp["P"], mdp["mu"])
e3, a, b = eta3(Phi, mdp["P"], mdp["mu"])
print("cond(A)=", condA, "eta1=%.4e eta2=%.4e eta3=%.4e ratio=%.3f" % (e1, e2, e3, e1 / e3))
print("W*==theta* ?", np.allclose(W_star, theta_star_v, atol=1e-9), "||W*||=", norm(W_star))

# double-chain, iid, constant stepsize alpha = eta1/18 (Theorem 4.1 bound)
eta = e1
alpha = eta / 18.0
T = 20000
theta0 = np.zeros(mdp["n"])
grid, err2 = run_double_chain(mdp, Phi, theta0, alpha, T, "iid", seed=1)
c_fit, floor = fit_decay_rate(grid, err2)
print("alpha=%.3e  alpha*eta=%.3e  c_fit=%.3e  ratio=%.2f" % (alpha, alpha * eta, c_fit, c_fit / (alpha * eta)))
print("err2[0]=%.3e  err2[-1]=%.3e  T_to_1e-2=%.0f" % (err2[0], err2[-1], t_to_eps(grid, err2, 1e-2)))
print("final theta close to theta*?", np.sqrt(err2[-1]) < 0.05 * norm(theta_star_v))

# mutation: correlated (use_indep=False)
grid2, err2b = run_double_chain(mdp, Phi, theta0, alpha, T, "iid", seed=1, use_indep=False)
print("MUTATION(correlated) err2[-1]=%.3e (should stay large -> property breaks)" % err2b[-1])
