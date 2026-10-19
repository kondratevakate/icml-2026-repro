"""Cheating-oracle lambda: pick the constant simplex weight vector w (K=3) that
MINIMISES the realised average MDCP set size ON THE TEST DATA, subject to
worst-case coverage >= 0.89. Upper bound on what any constant lambda can give.
Scale of lambda is irrelevant (score is monotone), so only the direction matters."""
import numpy as np, json, sys
import verify_mdrcp as V
from verify_mdrcp import K, ALPHA, NK, conf_p, spline_features

W = np.array([[a/10, b/10, (10-a-b)/10] for a in range(11) for b in range(11-a)])


def oracle_clf(seed):
    rng = np.random.default_rng(seed)
    Xs, Ys = V.gen_clf(rng)
    tr, ca, te = [], [], []
    for k in range(K):
        idx = rng.permutation(NK); n1, n2 = int(.375*NK), int(.5*NK)
        tr.append((Xs[k][idx[:n1]], Ys[k][idx[:n1]]))
        ca.append((Xs[k][idx[n1:n2]], Ys[k][idx[n1:n2]]))
        te.append((Xs[k][idx[n2:]], Ys[k][idx[n2:]]))
    mods = [V.softmax_fit(*t, 6) for t in tr]
    Pca = [np.stack([V.softmax_pred(m, ca[k][0]) for m in mods], 2) for k in range(K)]
    Pte = [np.stack([V.softmax_pred(m, te[k][0]) for m in mods], 2) for k in range(K)]
    # baseline max-p
    cal_tps = [-Pca[k][np.arange(len(ca[k][1])), ca[k][1], k] for k in range(K)]
    agg = []
    for k in range(K):
        pv = np.stack([conf_p(cal_tps[j], -Pte[k][:, :, j]) for j in range(K)], 0).max(0)
        agg.append((pv >= ALPHA).sum(1).mean())
    agg = float(np.mean(agg))
    best = (1e9, None)
    for w in W:
        cal = [-(Pca[k] * w).sum(2)[np.arange(len(ca[k][1])), ca[k][1]] for k in range(K)]
        sz, cv = [], []
        for k in range(K):
            s = -(Pte[k] * w).sum(2)
            pv = np.stack([conf_p(cal[j], s) for j in range(K)], 0).max(0)
            inc = pv >= ALPHA
            sz.append(inc.sum(1).mean())
            cv.append(inc[np.arange(len(te[k][1])), te[k][1]].mean())
        if min(cv) >= 0.89 and np.mean(sz) < best[0]:
            best = (float(np.mean(sz)), float(np.min(cv)))
    return agg, best


if __name__ == "__main__":
    rows = [oracle_clf(s) for s in range(10)]
    red = [100*(a-b[0])/a for a, b in rows]
    print("CLF oracle-lambda reduction vs max-p:", [round(r, 2) for r in red],
          "mean", round(float(np.mean(red)), 2), "sd", round(float(np.std(red, ddof=1)), 2),
          "worstcov", round(float(np.mean([b[1] for _, b in rows])), 4), flush=True)
    json.dump({"red": red}, open("res_oracle_clf.json", "w"))
