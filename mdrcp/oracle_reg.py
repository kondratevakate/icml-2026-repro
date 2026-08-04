"""Same cheating oracle for regression: constant simplex weights, chosen on test."""
import numpy as np, json
import verify_mdrcp as V
from verify_mdrcp import K, ALPHA, NK, conf_p, qhat
from oracle import W


def oracle_reg(seed, M=100):
    rng = np.random.default_rng(seed)
    Xs, Ys = V.gen_reg(rng)
    tr, ca, te = [], [], []
    for k in range(K):
        idx = rng.permutation(NK); n1, n2 = int(.375*NK), int(.5*NK)
        tr.append((Xs[k][idx[:n1]], Ys[k][idx[:n1]]))
        ca.append((Xs[k][idx[n1:n2]], Ys[k][idx[n1:n2]]))
        te.append((Xs[k][idx[n2:]], Ys[k][idx[n2:]]))
    mods = [V.fit_ms(*t) for t in tr]
    ally = np.concatenate([t[1] for t in tr] + [c[1] for c in ca])
    grid = np.linspace(ally.min(), ally.max(), M); dy = grid[1] - grid[0]

    def dens(Xa, yv):
        out = np.empty(yv.shape + (K,))
        for k in range(K):
            mu, sg = V.ms_pred(mods[k], Xa)
            z = (yv - mu[:, None]) / sg[:, None]
            out[:, :, k] = np.exp(-0.5*z**2) / (np.sqrt(2*np.pi)*sg[:, None])
        return out

    Dca = [dens(ca[k][0], ca[k][1][:, None])[:, 0, :] for k in range(K)]
    Dte = [dens(te[k][0], np.repeat(grid[None, :], len(te[k][1]), 0)) for k in range(K)]
    Dtey = [dens(te[k][0], te[k][1][:, None])[:, 0, :] for k in range(K)]
    # baseline max-p union length
    qs = [qhat(np.abs(ca[k][1]-V.ms_pred(mods[k], ca[k][0])[0])/V.ms_pred(mods[k], ca[k][0])[1])
          for k in range(K)]
    agg = []
    for k in range(K):
        lo = np.empty((len(te[k][1]), K)); hi = np.empty_like(lo)
        for j in range(K):
            mu, sg = V.ms_pred(mods[j], te[k][0])
            lo[:, j] = mu - qs[j]*sg; hi[:, j] = mu + qs[j]*sg
        g = grid[None, None, :]
        agg.append(((g >= lo[:, :, None]) & (g <= hi[:, :, None])).any(1).sum(1).mean()*dy)
    agg = float(np.mean(agg))
    best = (1e9, None)
    for w in W:
        cal = [-(Dca[k]*w).sum(1) for k in range(K)]
        sz, cv = [], []
        for k in range(K):
            pv = np.stack([conf_p(cal[j], -(Dte[k]*w).sum(2)) for j in range(K)], 0).max(0)
            sz.append((pv >= ALPHA).sum(1).mean()*dy)
            pvy = np.stack([conf_p(cal[j], -(Dtey[k]*w).sum(1)) for j in range(K)], 0).max(0)
            cv.append((pvy >= ALPHA).mean())
        if min(cv) >= 0.89 and np.mean(sz) < best[0]:
            best = (float(np.mean(sz)), float(np.min(cv)))
    return agg, best


if __name__ == "__main__":
    rows = [oracle_reg(s) for s in range(10)]
    red = [100*(a-b[0])/a for a, b in rows]
    print("REG oracle-lambda reduction vs max-p:", [round(r, 2) for r in red],
          "mean", round(float(np.mean(red)), 2), "sd", round(float(np.std(red, ddof=1)), 2),
          "worstcov", round(float(np.mean([b[1] for _, b in rows])), 4), flush=True)
