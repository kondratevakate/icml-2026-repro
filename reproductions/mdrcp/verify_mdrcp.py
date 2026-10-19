"""MDCP reproduction. Spec transcribed in spec.md (arXiv 2601.02998, Sec 5.1-5.3).
numpy+torch only, no scipy/sklearn."""
import numpy as np, torch, math
torch.set_num_threads(4)

ALPHA = 0.1
K, D, TAU, NK = 3, 10, 2.5, 2000


def make_X(rng, n):
    S = np.full((D, D), 0.2) + np.eye(D) * 0.8
    L = np.linalg.cholesky(S)
    return rng.standard_normal((n, D)) @ L.T


def bspline_basis(x, knots, deg=3):
    """Cox-de Boor, open uniform knot vector. x:(n,), returns (n, n_knots+deg-1)."""
    t = np.concatenate([np.repeat(knots[0], deg), knots, np.repeat(knots[-1], deg)])
    m = len(t) - deg - 1
    B = np.zeros((len(x), len(t) - 1))
    for i in range(len(t) - 1):
        B[:, i] = ((x >= t[i]) & (x < t[i + 1])).astype(float)
    B[(x >= t[-1]), -1] = 1.0
    for p in range(1, deg + 1):
        Bn = np.zeros((len(x), len(t) - p - 1))
        for i in range(len(t) - p - 1):
            d1 = t[i + p] - t[i]
            d2 = t[i + p + 1] - t[i + 1]
            a = (x - t[i]) / d1 * B[:, i] if d1 > 0 else 0.0
            b = (t[i + p + 1] - x) / d2 * B[:, i + 1] if d2 > 0 else 0.0
            Bn[:, i] = a + b
        B = Bn
    return B[:, :m]


def spline_features(Xtr, Xall_list, n_knots=5, deg=3):
    """Coordinatewise cubic B-spline (5 knots over observed range), concatenated + intercept."""
    outs = [[] for _ in Xall_list]
    for j in range(D):
        lo, hi = Xtr[:, j].min(), Xtr[:, j].max()
        kn = np.linspace(lo, hi, n_knots)
        for a, Xa in enumerate(Xall_list):
            xc = np.clip(Xa[:, j], lo, hi)
            outs[a].append(bspline_basis(xc, kn, deg))
    return [np.hstack(o + [np.ones((o[0].shape[0], 1))]) for o in outs]


def conf_p(cal_scores, test_scores):
    """p^(k)(y) = (1 + #{cal <= s}) / (1 + n_cal). test_scores any shape."""
    cs = np.sort(cal_scores)
    n = len(cs)
    # standard conformal p-value: fraction of calibration scores at least as
    # nonconforming as the test score (paper's eq (7), orientation fixed so that
    # low-nonconformity y are INCLUDED).
    cnt = n - np.searchsorted(cs, test_scores, side="left")
    return (1.0 + cnt) / (1.0 + n)


def fit_lambda(Phi_tr, P_tr, ppool_tr, alpha=ALPHA, epochs=400, lr=0.05):
    """Maximize (13): mean[min(1-h,0)/ppool] + (1-alpha)*mean[sum_k lambda_k].
    Phi_tr:(n,m) spline feats; P_tr:(n,K) = phat_k(Y_i|X_i); ppool_tr:(n,)."""
    Phi = torch.tensor(Phi_tr, dtype=torch.float32)
    P = torch.tensor(P_tr, dtype=torch.float32)
    pp = torch.tensor(ppool_tr, dtype=torch.float32).clamp_min(1e-8)
    Th = torch.zeros(Phi.shape[1], K, requires_grad=True)
    opt = torch.optim.Adam([Th], lr=lr)
    best, bestTh = -1e18, None
    for e in range(epochs):
        opt.zero_grad()
        lam = torch.nn.functional.softplus(Phi @ Th)          # (n,K)
        h = (lam * P).sum(1)
        obj = torch.clamp(1 - h, max=0.0).div(pp).mean() + (1 - alpha) * lam.sum(1).mean()
        (-obj).backward()
        opt.step()
        v = obj.item()
        if v > best:
            best, bestTh = v, Th.detach().clone()
    return bestTh.numpy()


def lam_of(Phi, Th):
    z = Phi @ Th
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


# ---------------- classification ----------------
C = 6

def gen_clf(rng):
    I = rng.choice(D, 4, replace=False)
    u = rng.uniform(-1, 1, K)
    xi = 2.5 * (1 + 0.25 * TAU * u)
    b = rng.normal(0, 0.4 * TAU, (K, C))
    bb = np.zeros((C, D)); bb[:, I] = rng.normal(0, 1, (C, 4))
    dl = np.zeros((K, C, D)); dl[:, :, I] = rng.normal(0, 0.15, (K, C, 4))
    beta = bb[None] + TAU * dl                                   # (K,C,D)
    Xs, Ys = [], []
    for k in range(K):
        X = make_X(rng, NK)
        eta = xi[k] * (b[k][None, :] + X @ beta[k].T)            # (n,C)
        pr = np.exp(eta - eta.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        cum = pr.cumsum(1)
        Y = (rng.random((NK, 1)) > cum).sum(1)
        Xs.append(X); Ys.append(Y)
    return Xs, Ys


def softmax_fit(X, Y, ncls, epochs=300, lr=0.1, l2=1e-3):
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.long)
    W = torch.zeros(X.shape[1], ncls, requires_grad=True); b = torch.zeros(ncls, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, Yt) + l2 * (W * W).sum()
        loss.backward(); opt.step()
    return W.detach().numpy(), b.detach().numpy()


def softmax_pred(Wb, X):
    W, b = Wb
    z = X @ W + b
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def run_clf(seed):
    rng = np.random.default_rng(seed)
    Xs, Ys = gen_clf(rng)
    tr, ca, te = [], [], []
    for k in range(K):
        idx = rng.permutation(NK); n1, n2 = int(.375 * NK), int(.5 * NK)
        tr.append((Xs[k][idx[:n1]], Ys[k][idx[:n1]]))
        ca.append((Xs[k][idx[n1:n2]], Ys[k][idx[n1:n2]]))
        te.append((Xs[k][idx[n2:]], Ys[k][idx[n2:]]))
    Xtr = np.vstack([t[0] for t in tr]); Ytr = np.concatenate([t[1] for t in tr])
    mods = [softmax_fit(t[0], t[1], C) for t in tr]
    pool = softmax_fit(Xtr, Ytr, C)

    Pt = np.stack([softmax_pred(m, Xtr) for m in mods], 2)       # (n,C,K)
    ii = np.arange(len(Ytr))
    Ptr_y = Pt[ii, Ytr, :]                                       # (n,K)
    ppool_y = softmax_pred(pool, Xtr)[ii, Ytr]

    Xall = [Xtr] + [c[0] for c in ca] + [t[0] for t in te]
    F = spline_features(Xtr, Xall)
    Th = fit_lambda_exact(F[0], Pt, epochs=3000)

    # --- per-source calib/test quantities
    def scores_mdcp(Xa, Phi):
        lam = lam_of(Phi, Th)                                     # (n,K)
        P = np.stack([softmax_pred(m, Xa) for m in mods], 2)      # (n,C,K)
        return -(P * lam[:, None, :]).sum(2), P                   # (n,C)

    cal_md, cal_tps, te_md, te_tps, te_Y = [], [], [], [], []
    for k in range(K):
        Xa, Ya = ca[k]; s, P = scores_mdcp(Xa, F[1 + k])
        cal_md.append(s[np.arange(len(Ya)), Ya])
        cal_tps.append(-P[np.arange(len(Ya)), Ya, k])
    for k in range(K):
        Xa, Ya = te[k]; s, P = scores_mdcp(Xa, F[1 + K + k])
        te_md.append(s); te_tps.append(P); te_Y.append(Ya)

    res = {}
    # MDCP: shared score, per-source p-values
    for name in ("MDCP", "AGG", "SRC"):
        covs, sizes, hits = [], [], []
        for k in range(K):
            Ya = te_Y[k]; n = len(Ya)
            if name == "MDCP":
                pv = np.stack([conf_p(cal_md[j], te_md[k]) for j in range(K)], 0).max(0)
                inset = pv >= ALPHA
            elif name == "AGG":
                pv = np.stack([conf_p(cal_tps[j], -te_tps[k][:, :, j]) for j in range(K)], 0).max(0)
                inset = pv >= ALPHA
            else:
                inset = conf_p(cal_tps[k], -te_tps[k][:, :, k]) >= ALPHA
            covs.append(inset[np.arange(n), Ya].mean())
            sizes.append(inset.sum(1).mean())
        res[name] = (float(np.mean(covs)), float(np.min(covs)), float(np.mean(sizes)))
    return res


def fit_lambda_exact(Phi_tr, Pfull, alpha=ALPHA, epochs=1500, lr=0.05):
    """Same dual objective (13), but the inner E_pool[.../phat_pool] is evaluated
    exactly as E_X[sum_y (1-h(x,y))_-] instead of by importance weighting.
    Pfull: (n, C, K) = phat_k(y|x). Maximize."""
    P = torch.tensor(Pfull, dtype=torch.float32)
    Phi = torch.tensor(Phi_tr, dtype=torch.float32)
    Th = torch.zeros(Phi.shape[1], K, requires_grad=True)
    opt = torch.optim.Adam([Th], lr=lr)
    best, bTh = -1e18, None
    for _ in range(epochs):
        opt.zero_grad()
        lam = torch.nn.functional.softplus(Phi @ Th)                # (n,K)
        h = (P * lam[:, None, :]).sum(2)                            # (n,C)
        obj = torch.clamp(1 - h, max=0.0).sum(1).mean() + (1 - alpha) * lam.sum(1).mean()
        (-obj).backward(); opt.step()
        if obj.item() > best:
            best, bTh = obj.item(), Th.detach().clone()
    return bTh.numpy()


# ---------------- regression ----------------
def gen_reg(rng):
    I = rng.choice(D, 4, replace=False)
    bb = np.zeros(D); bb[I] = rng.normal(0, 1, 4)
    dl = np.zeros((K, D)); dl[:, I] = rng.normal(0, 1, (K, 4))
    beta = bb[None] + 0.2 * TAU * dl
    b = rng.normal(0, 0.5) + TAU * rng.normal(0, 0.5, K)
    snr = rng.uniform(5, 10)
    Xs, Ys = [], []
    for k in range(K):
        X = make_X(rng, NK)
        m = X @ beta[k] + b[k]
        sd = np.sqrt(m.var() / snr)
        Xs.append(X); Ys.append(m + rng.normal(0, sd, NK))
    return Xs, Ys


def ridge(X, y, l2=1e-3):
    A = np.hstack([X, np.ones((len(X), 1))])
    W = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ y)
    return W


def rpred(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def fit_ms(X, y):
    """mu via ridge; sigma via 5-fold out-of-fold log squared-residual regression (C.2)."""
    W = ridge(X, y)
    n = len(y); idx = np.arange(n); oof = np.zeros(n)
    for f in range(5):
        m = idx % 5 == f
        oof[m] = rpred(ridge(X[~m], y[~m]), X[m])
    lr2 = np.log(np.maximum((y - oof) ** 2, 1e-8))
    Wv = ridge(X, lr2)
    return W, Wv


def ms_pred(M, X):
    W, Wv = M
    return rpred(W, X), np.sqrt(np.exp(np.clip(rpred(Wv, X), -20, 20))) + 1e-6


def fit_lambda_reg(Phi, F, dy, alpha=ALPHA, epochs=4000, lr=0.05):
    """F: (n, M, K) densities on the y-grid; dy grid spacing."""
    Ft = torch.tensor(F, dtype=torch.float32); P = torch.tensor(Phi, dtype=torch.float32)
    Th = torch.zeros(Phi.shape[1], K, requires_grad=True)
    opt = torch.optim.Adam([Th], lr=lr)
    best, bTh = -1e18, None
    for _ in range(epochs):
        opt.zero_grad()
        lam = torch.nn.functional.softplus(P @ Th)
        h = (Ft * lam[:, None, :]).sum(2)
        obj = (torch.clamp(1 - h, max=0.0).sum(1) * dy).mean() + (1 - alpha) * lam.sum(1).mean()
        (-obj).backward(); opt.step()
        if obj.item() > best:
            best, bTh = obj.item(), Th.detach().clone()
    return bTh.numpy()


def qhat(scores, alpha=ALPHA):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.sort(scores)[min(k, n) - 1]


def run_reg(seed, M=100):
    rng = np.random.default_rng(seed)
    Xs, Ys = gen_reg(rng)
    tr, ca, te = [], [], []
    for k in range(K):
        idx = rng.permutation(NK); n1, n2 = int(.375 * NK), int(.5 * NK)
        tr.append((Xs[k][idx[:n1]], Ys[k][idx[:n1]]))
        ca.append((Xs[k][idx[n1:n2]], Ys[k][idx[n1:n2]]))
        te.append((Xs[k][idx[n2:]], Ys[k][idx[n2:]]))
    Xtr = np.vstack([t[0] for t in tr]); Ytr = np.concatenate([t[1] for t in tr])
    mods = [fit_ms(*t) for t in tr]
    ally = np.concatenate([Ytr] + [c[1] for c in ca])
    yL, yU = ally.min(), ally.max(); grid = np.linspace(yL, yU, M); dy = grid[1] - grid[0]

    def dens(Xa):
        out = np.empty((len(Xa), M, K))
        for k in range(K):
            mu, sg = ms_pred(mods[k], Xa)
            z = (grid[None, :] - mu[:, None]) / sg[:, None]
            out[:, :, k] = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sg[:, None])
        return out

    Xall = [Xtr] + [c[0] for c in ca] + [t[0] for t in te]
    F = spline_features(Xtr, Xall)
    Th = fit_lambda_reg(F[0], dens(Xtr), dy)

    def s_md(Xa, Phi, yv):
        """MDCP score at arbitrary y values yv (n,Q)."""
        lam = lam_of(Phi, Th)
        h = np.zeros(yv.shape)
        for k in range(K):
            mu, sg = ms_pred(mods[k], Xa)
            z = (yv - mu[:, None]) / sg[:, None]
            h += lam[:, [k]] * np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sg[:, None])
        return -h

    cal_md = [s_md(ca[k][0], F[1 + k], ca[k][1][:, None])[:, 0] for k in range(K)]
    cal_ab = [np.abs(ca[k][1] - ms_pred(mods[k], ca[k][0])[0]) / ms_pred(mods[k], ca[k][0])[1]
              for k in range(K)]
    qs = [qhat(c) for c in cal_ab]

    res = {n: [[], []] for n in ("MDCP", "AGG", "SRC")}
    for k in range(K):
        Xa, Ya = te[k]; n = len(Ya)
        # baselines: per-source intervals, union length
        lo = np.empty((n, K)); hi = np.empty((n, K))
        for j in range(K):
            mu, sg = ms_pred(mods[j], Xa)
            lo[:, j] = mu - qs[j] * sg; hi[:, j] = mu + qs[j] * sg
        inU = ((Ya[:, None] >= lo) & (Ya[:, None] <= hi)).any(1)
        # union length via grid (same grid resolution for fairness)
        gv = grid[None, None, :]
        covg = ((gv >= lo[:, :, None]) & (gv <= hi[:, :, None])).any(1)
        res["AGG"][0].append(inU.mean()); res["AGG"][1].append(covg.sum(1).mean() * dy)
        res["SRC"][0].append(((Ya >= lo[:, 0]) & (Ya <= hi[:, 0])).mean())
        res["SRC"][1].append((hi[:, 0] - lo[:, 0]).mean())
        # MDCP grid search
        sg_ = s_md(Xa, F[1 + K + k], np.repeat(grid[None, :], n, 0))
        pv = np.stack([conf_p(cal_md[j], sg_) for j in range(K)], 0).max(0)
        inc = pv >= ALPHA
        # block-merge + extend by dy  => length = (#included + #blocks*2) * dy, clipped
        # plain grid Lebesgue measure, same convention used for the baseline union
        length = inc.sum(1) * dy
        sy = s_md(Xa, F[1 + K + k], Ya[:, None])[:, 0]
        pvy = np.stack([conf_p(cal_md[j], sy) for j in range(K)], 0).max(0)
        res["MDCP"][0].append((pvy >= ALPHA).mean()); res["MDCP"][1].append(length.mean())
    return {n: (float(np.mean(v[0])), float(np.min(v[0])), float(np.mean(v[1])))
            for n, v in res.items()}
