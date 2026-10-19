"""PAPER-DERIVED reimplementation of CAME-Grad (arXiv:2605.22635, Sec. 3.3, Algorithm 1, Eqs. 8-14).

This is NOT the authors' code: the repository withholds modules/CAME_Grad.py (README TODO).
Only the call signature is taken from the repository (modules/trainer.py):
    CAME_Grad(optimizer, rho=..., kappa=..., nu=...)
    optimizer.step(losses, task_weights=[...])
Everything else follows the paper's equations:
    mu      = mean_i g_i                                              (Alg. 1)
    g_joint = sum_i w_i g_i                                           (Eq. 4)
    Stage 1: alpha* = argmin_{simplex} g_alpha^T mu + sqrt(xi)||g_alpha||,  xi = rho^2 ||mu||^2   (Eq. 9)
             u_rect = mu + sqrt(xi)/||g_alpha*|| * g_alpha*                                        (Eq. 10)
    Stage 2: tau = kappa ||g_joint||;  u_en = u_rect * tau / (||u_rect|| + eps)                    (Eqs. 11-12)
    Stage 3: g_final = (1-nu) u_en + nu * kappa * g_joint                                          (Eq. 13)
    update : theta <- theta - eta * g_final (here: g_final is written to .grad and the wrapped
             optimizer's own step is applied, as the trainer wraps a torch AdamW)

Ablation conventions (the paper does not define them; these are this check's choices):
    S1 off: u_rect := g_joint      S2 off: u_en := u_rect      S3 off: nu := 0
"""
import numpy as np
import torch
from scipy.optimize import minimize


def project_simplex(v):
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, 0) - 1.0
    ind = torch.arange(1, n + 1, dtype=v.dtype)
    cond = (u - css / ind) > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return torch.clamp(v - theta, min=0.0)


def dual_value(alpha, M, b, c):
    q = alpha @ M @ alpha
    return alpha @ b + c * torch.sqrt(q + 1e-30)


def solve_dual_torch(M, b, c, iters=5000, tol=1e-15):
    """Tensorised projected-gradient solver (Armijo backtracking) for Eq. 9 on the simplex."""
    n = b.numel()
    alpha = torch.full((n,), 1.0 / n, dtype=b.dtype)
    f = dual_value(alpha, M, b, c)
    step = 1.0
    for _ in range(iters):
        q = alpha @ M @ alpha
        g = b + c * (M @ alpha) / torch.sqrt(q + 1e-30)
        while True:
            new = project_simplex(alpha - step * g)
            fn = dual_value(new, M, b, c)
            if fn <= f + 1e-4 * (g @ (new - alpha)) or step < 1e-18:
                break
            step *= 0.5
        moved = (new - alpha).abs().max().item()
        alpha, f = new, fn
        step *= 1.5
        if moved < tol:
            break
    return alpha


def solve_dual_slsqp(M, b, c):
    """Independent solver in the style of CAGrad's official code (scipy SLSQP)."""
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(b)
    fun = lambda a: float(a @ b + c * np.sqrt(a @ M @ a + 1e-30))
    res = minimize(fun, np.ones(n) / n, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=({"type": "eq", "fun": lambda a: a.sum() - 1.0},),
                   options={"maxiter": 2000, "ftol": 1e-15})
    return torch.tensor(res.x, dtype=torch.float64), float(res.fun)


def stage1_rectify(G, rho, solver="torch"):
    """G: (n, d) float64 task gradients. Returns u_rect, alpha*, mu, dual value."""
    mu = G.mean(0)
    xi = (rho * mu.norm()) ** 2
    c = torch.sqrt(xi)
    M = G @ G.T
    b = G @ mu
    if solver == "torch":
        alpha = solve_dual_torch(M, b, c)
    else:
        alpha, _ = solve_dual_slsqp(M.numpy(), b.numpy(), float(c))
    g_alpha = alpha @ G
    u = mu + c / (g_alpha.norm() + 1e-12) * g_alpha
    return u, alpha, mu, float(dual_value(alpha, M, b, c))


def came_direction(G, w, rho, kappa, nu, eps=1e-8, stages=(True, True, True), solver="torch"):
    """Full cascade; returns all intermediates (float64)."""
    G = G.to(torch.float64)
    w = torch.as_tensor(w, dtype=torch.float64)
    s1, s2, s3 = stages
    g_joint = w @ G
    mu = G.mean(0)
    if s1:
        u_rect, alpha, mu, dval = stage1_rectify(G, rho, solver)
    else:
        u_rect, alpha, dval = g_joint.clone(), None, None
    tau = kappa * g_joint.norm()
    u_en = u_rect * (tau / (u_rect.norm() + eps)) if s2 else u_rect.clone()
    nu_eff = nu if s3 else 0.0
    g_final = (1.0 - nu_eff) * u_en + nu_eff * kappa * g_joint
    return dict(g_joint=g_joint, mu=mu, u_rect=u_rect, u_en=u_en, g_final=g_final, alpha=alpha,
                dual=dval, tau=tau)


class CAME_Grad:
    """Drop-in wrapper with the interface used by the repository's trainer."""

    def __init__(self, optimizer, rho=0.5, kappa=1.5, nu=0.2, eps=1e-8, stages=(True, True, True),
                 solver="torch"):
        self.optimizer = optimizer
        self.rho, self.kappa, self.nu, self.eps = rho, kappa, nu, eps
        self.stages, self.solver = stages, solver
        self.last = None

    # the repo's LR scheduler mutates optimizer.param_groups[*]["lr"]
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def _params(self):
        return [p for g in self.optimizer.param_groups for p in g["params"] if p.requires_grad]

    def step(self, losses, task_weights=None):
        params = self._params()
        if task_weights is None:
            task_weights = [1.0] * len(losses)
        rows = []
        used = [False] * len(params)   # params receiving no gradient from any task keep p.grad = None,
        for L in losses:               # exactly as in the plain backward() path (AdamW then skips them)
            gs = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
            for j, g in enumerate(gs):
                used[j] = used[j] or (g is not None)
            rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                   for g, p in zip(gs, params)]))
        G = torch.stack(rows)
        out = came_direction(G, task_weights, self.rho, self.kappa, self.nu, self.eps,
                             self.stages, self.solver)
        gf = out["g_final"].to(params[0].dtype)
        off = 0
        for j, p in enumerate(params):
            n = p.numel()
            p.grad = gf[off:off + n].view_as(p).clone() if used[j] else None
            off += n
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.last = {k: (v if not torch.is_tensor(v) else v.detach()) for k, v in out.items()}
        self.last["G_cos"] = None
        return out
