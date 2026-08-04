"""
Independent reproduction of:
  "A Solvable High-Dimensional Model Where Nonlinear Autoencoders Learn Structure
   Invisible to PCA While Test Loss Misaligns With Generalization"
  Conde Mendes, Bardone, Koller, Medina Moreira, Erba, Troiani, Zdeborova.
  ICML 2026 (orid wm3ABfhE7P), arXiv 2602.10680.

No code was released. This is a from-the-paper reimplementation: every constant,
distribution, architecture and hyperparameter is transcribed from the paper, not
invented. Equation / table / section numbers refer to the arXiv PDF.

Challenge claims tested (2 claims):
  C1: A minimal nonlinear autoencoder recovers BOTH latent spikes (u* and the hidden
      v*), while PCA / linear autoencoders recover only u*.  -> theta_v(nonlinear) > floor,
      theta_v(linear/PCA) ~ floor, with the ReLU<->tanh flip between k*=2 and k*=3 (Table 1).
  C2: Self-supervised test (reconstruction) loss misaligns with representation quality:
      the linear AE has LOWER test reconstruction loss yet HIGHER downstream error than
      the nonlinear AE which recovers v* (Fig. 2).

Data-generating spiked-cumulant model (Eq. 1, Eq. 2):
  x = lam * u*/sqrt(d) + S ( nu * v*/sqrt(d) + z ),   u*,v*,z ~ N(0, I_d)
  S = I - [ E[nu^2] / (1 + E[nu^2] + sqrt(1 + E[nu^2])) ] * v* v*^T / d
  Latents (E[lam]=E[nu]=0, E[lam*nu]=0):
    k*=2 (Fig. 1 / Sec. E):   lam ~ N(0,1),  nu = -sqrt(2) if |lam| < Phi^{-1}(0.75) else +sqrt(2)   (E[nu^2]=2)
    k*=3 (Sec. E.4):          lam ~ N(0,1),  nu = sign(lam) * sign(|lam| - sqrt(2 ln 2))            (E[nu^2]=1)

Minimal nonlinear autoencoder (Eq. 4), single neuron, tied weights, no bias:
  xhat = (w / sqrt(d)) * sigma( w^T x / sqrt(d) ),   loss = mean_mu || x_mu - xhat_mu ||^2  (Eq. 6)
  Trained with full-batch Adam, lr=0.1, no weight decay, init w ~ N(0, I_d)  (paper Appendix E).

Recovery metric: cosine similarity theta_s = |w . s| / (||w|| ||s||),  s in {u*, v*}.
Floor (independent random direction) ~ d^{-1/2}: below it, no weak recovery (paper Fig. 7).
"""
import argparse
import time
import numpy as np

# --- constants transcribed from the paper (no scipy needed) ---
PHI_INV_075 = 0.6744897501960817     # Phi^{-1}(0.75), threshold for the k*=2 latent law
SQRT_2LN2   = 1.1774100225154747     # sqrt(2 ln 2), threshold for the k*=3 latent law
SQRT2       = np.sqrt(2.0)


# ----------------------------- activations (sigma, sigma') -----------------------------
def _relu(a):    return np.maximum(a, 0.0), (a > 0.0).astype(a.dtype)
def _linear(a):  return a, np.ones_like(a)
def _tanh(a):    t = np.tanh(a); return t, 1.0 - t * t
def _quad(a):    return a * a, 2.0 * a
def _sigmoid(a): s = 1.0 / (1.0 + np.exp(-a)); return s, s * (1.0 - s)
def _elu(a):
    e = np.where(a > 0.0, a, np.exp(np.minimum(a, 0.0)) - 1.0)
    ep = np.where(a > 0.0, 1.0, np.exp(np.minimum(a, 0.0)))
    return e.astype(a.dtype), ep.astype(a.dtype)

ACTIVATIONS = {
    "linear": _linear,   # sigma = id  ->  provably equals PCA (Prop. 2.1)
    "relu": _relu,       # recovers v* at k*=2, FAILS at k*=3  (Table 1)
    "tanh": _tanh,       # FAILS at k*=2, recovers v* at k*=3  (Table 1)
    "quadratic": _quad,  # fails at both (C_2 >= 0, case x2)
    "sigmoid": _sigmoid,
    "elu": _elu,
}


# ------------------------------- data model (Eq. 1, Eq. 2) -----------------------------
def gen_latents(n, kstar, rng):
    """Return lam, nu, E[nu^2] for the paper's exact latent laws."""
    lam = rng.standard_normal(n)
    if kstar == 2:
        nu = np.where(np.abs(lam) < PHI_INV_075, -SQRT2, SQRT2)
        Enu2 = 2.0
    elif kstar == 3:
        nu = np.sign(lam) * np.sign(np.abs(lam) - SQRT_2LN2)
        Enu2 = 1.0
    else:
        raise ValueError("kstar must be 2 or 3")
    return lam.astype(np.float64), nu.astype(np.float64), Enu2


def gen_data(n, u, v, kstar, rng, dtype=np.float32):
    """x = lam u/sqrt(d) + S(nu v/sqrt(d) + z), with S the rank-1 whitening (Eq. 2)."""
    d = u.shape[0]
    sd = np.sqrt(d)
    lam, nu, Enu2 = gen_latents(n, kstar, rng)
    cS = Enu2 / (1.0 + Enu2 + np.sqrt(1.0 + Enu2))          # scalar in S (Eq. 2)
    z = rng.standard_normal((n, d)).astype(dtype)
    inner = (nu[:, None].astype(dtype) * v[None, :]) / sd + z   # nu v/sqrt(d) + z
    vdot = (inner @ v) / d                                   # (v . inner)/d  per sample
    S_inner = inner - cS * vdot[:, None].astype(dtype) * v[None, :]  # S applied row-wise
    X = (lam[:, None].astype(dtype) * u[None, :]) / sd + S_inner
    return X.astype(dtype), lam, nu


# --------------------- autoencoder loss + analytic gradient (Eq. 4, Eq. 6) -------------
def ae_loss_grad(w, X, act):
    """
    xhat = (w/sqrt(d)) sigma(a),  a = X w / sqrt(d).
    dL/dw = mean_mu -2[ (g_mu/sqrt(d)) R_mu + sigma'(a_mu) (R_mu . w)/d  x_mu ],  R = x - xhat.
    """
    n, d = X.shape
    sd = np.sqrt(d)
    sigma, _ = None, None
    a = (X @ w) / sd                          # (n,)
    g, gp = act(a)                            # sigma(a), sigma'(a)
    Xhat = (g[:, None] * w[None, :]) / sd      # (n,d)
    R = X - Xhat
    loss = float(np.mean(np.sum(R * R, axis=1)))
    rw = R @ w                                # (n,) = R_mu . w
    term1 = (g / sd)[:, None] * R
    term2 = (gp * rw / d)[:, None] * X
    grad = -2.0 * np.mean(term1 + term2, axis=0)
    return loss, grad.astype(w.dtype)


def train_ae(X, act, epochs, lr, w0):
    """Full-batch Adam (lr=0.1, betas=(0.9,0.999), eps=1e-8), no weight decay (paper App. E)."""
    w = w0.copy()
    m = np.zeros_like(w); vsq = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    last = None
    for t in range(1, epochs + 1):
        loss, grad = ae_loss_grad(w, X, act)
        m = b1 * m + (1 - b1) * grad
        vsq = b2 * vsq + (1 - b2) * (grad * grad)
        mhat = m / (1 - b1 ** t)
        vhat = vsq / (1 - b2 ** t)
        w = w - lr * mhat / (np.sqrt(vhat) + eps)
        last = loss
    return w, last


# ----------------------------------- metrics ------------------------------------------
def cosine(w, s):
    return float(abs(w @ s) / (np.linalg.norm(w) * np.linalg.norm(s) + 1e-30))


def pca_top(X):
    """Top principal direction (right singular vector of centered X) -> Prop. 2.1 baseline."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD; top right singular vector
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[0]


def test_recon_mse(w, X, act):
    n, d = X.shape
    sd = np.sqrt(d)
    a = (X @ w) / sd
    g, _ = act(a)
    R = X - (g[:, None] * w[None, :]) / sd
    return float(np.mean(np.sum(R * R, axis=1)))


# ------------------------------- experiment drivers -----------------------------------
def run_recovery(d, alpha, kstar, activations, instances, epochs, lr, seed, verbose=False):
    """Average theta_u, theta_v over instances for each activation, plus a PCA baseline."""
    floor = 1.0 / np.sqrt(d)
    n = int(alpha * d)
    acc = {name: {"theta_u": [], "theta_v": []} for name in activations}
    acc["PCA"] = {"theta_u": [], "theta_v": []}
    for inst in range(instances):
        rng = np.random.default_rng(seed + 1000 * inst + 7 * kstar)
        u = rng.standard_normal(d); v = rng.standard_normal(d)
        X, _, _ = gen_data(n, u, v, kstar, rng)
        w0 = rng.standard_normal(d).astype(X.dtype)      # init w ~ N(0, I_d)
        # PCA baseline (once per instance)
        pv = pca_top(X)
        acc["PCA"]["theta_u"].append(cosine(pv, u))
        acc["PCA"]["theta_v"].append(cosine(pv, v))
        for name in activations:
            w, _ = train_ae(X, ACTIVATIONS[name], epochs, lr, w0)
            acc[name]["theta_u"].append(cosine(w, u))
            acc[name]["theta_v"].append(cosine(w, v))
        if verbose:
            print(f"    instance {inst+1}/{instances} done")
    summary = {}
    for name, dct in acc.items():
        summary[name] = {
            "theta_u": float(np.mean(dct["theta_u"])),
            "theta_v": float(np.mean(dct["theta_v"])),
            "theta_v_std": float(np.std(dct["theta_v"])),
        }
    return summary, floor


def run_misalignment(d, alpha, kstar, epochs, lr, seed, n_down=200):
    """
    Claim 2: linear AE has LOWER test reconstruction loss but HIGHER downstream error
    than the nonlinear AE that recovers v*.  Downstream label y = sign(x . v*) (Fig. 2).
    Compares linear vs the recovering nonlinearity (relu for k*=2, tanh for k*=3).
    """
    nonlin = "relu" if kstar == 2 else "tanh"
    n = int(alpha * d)
    rng = np.random.default_rng(seed + 99)
    u = rng.standard_normal(d); v = rng.standard_normal(d)
    Xtr, _, _ = gen_data(n, u, v, kstar, rng)
    w0 = rng.standard_normal(d).astype(Xtr.dtype)
    Xte, _, _ = gen_data(20000, u, v, kstar, rng)
    yte = np.sign(Xte @ v)
    out = {}
    for name in ["linear", nonlin]:
        w, _ = train_ae(Xtr, ACTIVATIONS[name], epochs, lr, w0)
        test_loss = test_recon_mse(w, Xte, ACTIVATIONS[name])
        # downstream: linear predictor initialised from pre-trained w (sign of projection)
        yhat = np.sign(Xte @ w)
        err = float(np.mean(yhat != yte))
        err = min(err, 1.0 - err)            # sign of w is arbitrary; report the aligned error
        out[name] = {"test_recon_mse": test_loss, "downstream_err": err,
                     "theta_v": cosine(w, v)}
    return out, nonlin


# ------------------------------------- smoke test -------------------------------------
def smoke():
    """Rule 10: run the full pipeline on ONE tiny config, time it, estimate the full run."""
    print("=== SMOKE TEST (device: CPU / numpy) ===")
    t0 = time.time()
    d, alpha, kstar, epochs = 200, 5.0, 2, 200
    rng = np.random.default_rng(0)
    u = rng.standard_normal(d); v = rng.standard_normal(d)
    X, _, _ = gen_data(int(alpha * d), u, v, kstar, rng)
    w0 = rng.standard_normal(d).astype(X.dtype)
    w, loss = train_ae(X, ACTIVATIONS["relu"], epochs, 0.1, w0)
    tv = cosine(w, v); tu = cosine(w, u); floor = 1.0 / np.sqrt(d)
    dt = time.time() - t0
    assert np.isfinite(loss) and np.isfinite(tv), "smoke: non-finite output"
    print(f"  one relu train ({epochs} ep, d={d}, n={int(alpha*d)}): {dt:.2f}s  "
          f"loss={loss:.3f}  theta_u={tu:.3f}  theta_v={tv:.3f}  floor={floor:.3f}")
    print(f"  per-train ~{dt:.2f}s at d={d},ep={epochs}. Cost scales ~ n*d*epochs.")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=500)
    ap.add_argument("--alphas", type=float, nargs="+", default=[8.0, 20.0])
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--acts", type=str, nargs="+",
                    default=["linear", "relu", "tanh", "quadratic"])
    ap.add_argument("--kstars", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    smoke()
    if args.smoke_only:
        return

    print("\n=== RECOVERY SWEEP (Claim 1: nonlinear recovers v*, PCA/linear do not) ===")
    print(f"d={args.d}  epochs={args.epochs}  instances={args.instances}  "
          f"acts={args.acts}  alphas={args.alphas}")
    for kstar in args.kstars:
        print(f"\n--- k* = {kstar}  (predicted v*-recoverers: "
              f"{'relu,tanh(no),elu,sigmoid' if kstar==2 else 'tanh,sigmoid,elu; relu FAILS'}) ---")
        for alpha in args.alphas:
            summ, floor = run_recovery(args.d, alpha, kstar, args.acts,
                                       args.instances, args.epochs, args.lr, args.seed)
            print(f"  alpha={alpha:<5}  floor(theta~random)={floor:.3f}")
            for name in args.acts + ["PCA"]:
                s = summ[name]
                rec = "RECOVERS v*" if s["theta_v"] > 2 * floor else "no v*"
                print(f"    {name:<10} theta_u={s['theta_u']:.3f}  "
                      f"theta_v={s['theta_v']:.3f}+-{s['theta_v_std']:.3f}  [{rec}]")

    print("\n=== MISALIGNMENT (Claim 2: lower test loss, worse downstream) ===")
    for kstar in args.kstars:
        alpha = max(args.alphas)
        out, nonlin = run_misalignment(args.d, alpha, kstar, args.epochs, args.lr, args.seed)
        lin, nl = out["linear"], out[nonlin]
        print(f"  k*={kstar}, alpha={alpha}:")
        print(f"    linear : test_recon_mse={lin['test_recon_mse']:.4f}  "
              f"downstream_err={lin['downstream_err']:.3f}  theta_v={lin['theta_v']:.3f}")
        print(f"    {nonlin:<7}: test_recon_mse={nl['test_recon_mse']:.4f}  "
              f"downstream_err={nl['downstream_err']:.3f}  theta_v={nl['theta_v']:.3f}")
        misaligned = (lin["test_recon_mse"] < nl["test_recon_mse"]) and \
                     (lin["downstream_err"] > nl["downstream_err"])
        print(f"    -> misalignment reproduced: {misaligned} "
              f"(linear lower loss but worse downstream)")


if __name__ == "__main__":
    main()
