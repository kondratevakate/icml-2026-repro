"""
CalPro reproduction -- Phase 2: does an evidential (NIG) head + uncertainty-normalized
conformal RESTORE coverage under distribution shift, where vanilla split conformal (Phase 1)
collapses? This reproduces CalPro's central claim 1 on the paper's own non-biological
heteroscedastic-regression-under-shift setting (arXiv 2601.07201). CPU, torch.

Mechanism (from the paper):
  - Normal-Inverse-Gamma evidential head (Amini et al., 2020): MLP -> (gamma=mu, nu, alpha, beta),
    trained with the NIG NLL + an evidence regularizer sum exp(-alpha) that discourages
    overconfidence without data support.
  - Predictive uncertainty sigma(x) from the NIG posterior (aleatoric + epistemic). Out of the
    training region the epistemic term grows (nu -> small), so sigma(x) grows.
  - UNCERTAINTY-NORMALIZED nonconformity  s_i = |y_i - mu_i| / sigma_i  (adaptive conformal).
    Calibrate q_hat on normalized scores; interval half-width at x is q_hat * sigma(x). Under
    shift, sigma grows -> intervals widen where the model is uncertain -> coverage recovers.

Compared head-to-head against the Phase-1 vanilla split conformal (constant-width |y-mu|).
"""
import numpy as np
import torch
import torch.nn as nn

from verify_calpro import sample_data, split_conformal_interval, empirical_coverage

torch.manual_seed(0)
DEVICE = "cpu"


# ------------------------------- NIG evidential head ----------------------------------
class NIGHead(nn.Module):
    """MLP -> (mu, nu, alpha, beta) with positivity + alpha>1 for finite variance."""
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, 4)
        self.sp = nn.Softplus()

    def forward(self, x):
        h = self.body(x)
        mu, log_nu, log_alpha, log_beta = self.head(h).chunk(4, dim=-1)
        nu = self.sp(log_nu) + 1e-6
        alpha = self.sp(log_alpha) + 1.0 + 1e-6      # alpha > 1 => finite mean/var
        beta = self.sp(log_beta) + 1e-6
        return mu, nu, alpha, beta


def nig_nll(y, mu, nu, alpha, beta):
    """Amini et al. (2020) Normal-Inverse-Gamma negative log-likelihood."""
    om = 2.0 * beta * (1.0 + nu)                      # Omega
    nll = (0.5 * torch.log(np.pi / nu)
           - alpha * torch.log(om)
           + (alpha + 0.5) * torch.log(nu * (y - mu) ** 2 + om)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    return nll


def predictive_sigma(nu, alpha, beta):
    """Total predictive std = sqrt(aleatoric + epistemic) of the NIG posterior."""
    aleatoric = beta / (alpha - 1.0)                  # E[sigma^2]
    epistemic = beta / (nu * (alpha - 1.0))           # Var of the mean
    return torch.sqrt(aleatoric + epistemic)


def train_nig(xtr, ytr, epochs=1500, lr=5e-3, lam=1e-2):
    model = NIGHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(xtr, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    for ep in range(epochs):
        opt.zero_grad()
        mu, nu, alpha, beta = model(X)
        loss = nig_nll(Y, mu, nu, alpha, beta).mean() + lam * torch.exp(-alpha).mean()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def infer(model, x):
    X = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    mu, nu, alpha, beta = model(X)
    sigma = predictive_sigma(nu, alpha, beta)
    return mu.view(-1).numpy(), sigma.view(-1).numpy()


# --------------------------------------- run ------------------------------------------
def run(taus=(0.8, 0.9, 0.95)):
    rng = np.random.default_rng(0)
    xtr, ytr = sample_data(3000, -2.0, 2.0, rng)
    xcal, ycal = sample_data(3000, -2.0, 2.0, rng)
    xte, yte = sample_data(5000, -2.0, 2.0, rng)      # in-distribution
    xsh, ysh = sample_data(5000, 2.0, 3.5, rng)       # covariate shift

    model = train_nig(xtr, ytr)
    mu_cal, sig_cal = infer(model, xcal)
    mu_te, sig_te = infer(model, xte)
    mu_sh, sig_sh = infer(model, xsh)

    print(f"mean predictive sigma: in-dist={sig_te.mean():.3f}  shifted={sig_sh.mean():.3f}"
          f"  (ratio {sig_sh.mean()/sig_te.mean():.2f}x -> evidential head grows uncertainty OOD)")
    # NORMALIZED nonconformity: s = |y - mu| / sigma
    cal_scores = np.abs(ycal - mu_cal) / sig_cal
    print(f"\n{'tau':>5} {'q_hat':>7} {'cov_indist':>11} {'cov_shift':>10}  vs Phase-1 vanilla shift")
    for tau in taus:
        q = split_conformal_interval(cal_scores, tau)
        cov_id = empirical_coverage(yte, mu_te, q * sig_te)     # adaptive half-width q*sigma(x)
        cov_sh = empirical_coverage(ysh, mu_sh, q * sig_sh)
        print(f"{tau:5.2f} {q:7.3f} {cov_id:11.3f} {cov_sh:10.3f}   "
              f"(Phase-1 vanilla dropped to ~0.23-0.43)")
    print("\nClaim-1 reproduced if shifted coverage stays near nominal (evidential-normalized")
    print("conformal restores coverage where vanilla split conformal collapsed under shift).")


if __name__ == "__main__":
    run()
