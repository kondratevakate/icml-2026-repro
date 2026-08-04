"""ertlib.py — shared implementation of Algorithm 1 (ERT estimation) and baseline metrics.

Only numpy / scipy / scikit-learn are used. Losses follow Table 1 of the paper.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

EPS = 1e-6


# ---------------- proper scores (Table 1) ----------------
def loss(name, pred, z, t):
    pred = np.asarray(pred, dtype=float)
    z = np.asarray(z, dtype=float)
    if name == "L1":
        return np.sign(pred - t) * (t - z)
    if name == "L2":
        return (z - pred) ** 2
    if name == "KL":
        pr = np.clip(pred, EPS, 1 - EPS)
        return -(z * np.log(pr) + (1 - z) * np.log(1 - pr))
    raise ValueError(name)


def ert_terms(name, pred, z, t):
    """per-sample l(1-a, z) - l(pred, z)."""
    const = np.full_like(np.asarray(pred, dtype=float), t)
    return loss(name, const, z, t) - loss(name, pred, z, t)


# ---------------- classifiers ----------------
class PartitionWise:
    """The classifier implicitly underlying CovGap: k-means groups, predict group mean of Z."""

    def __init__(self, n_groups=10, seed=0):
        self.n_groups = n_groups
        self.seed = seed

    def fit(self, X, z):
        self.km = KMeans(n_clusters=self.n_groups, n_init=3, random_state=self.seed).fit(X)
        lab = self.km.labels_
        self.mu = np.array([z[lab == g].mean() if (lab == g).any() else z.mean()
                            for g in range(self.n_groups)])
        self.glob = z.mean()
        return self

    def predict_proba1(self, X):
        return self.mu[self.km.predict(X)]


class GBClassifier:
    """Gradient-boosted trees; LightGBM if available, else sklearn HistGradientBoosting."""

    def __init__(self, seed=0):
        self.seed = seed
        try:
            import lightgbm as lgb  # noqa
            self.kind = "lightgbm"
        except Exception:
            self.kind = "sklearn_hgb"

    def fit(self, X, z):
        if len(np.unique(z)) < 2:
            self.const = float(z.mean())
            self.model = None
            return self
        self.const = None
        if self.kind == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                            num_leaves=31, min_child_samples=20,
                                            n_jobs=2, random_state=self.seed,
                                            verbose=-1).fit(X, z)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, random_state=self.seed).fit(X, z)
        return self

    def predict_proba1(self, X):
        if self.model is None:
            return np.full(len(X), self.const)
        return self.model.predict_proba(X)[:, 1]


def make_classifier(kind, seed=0, n_groups=10):
    if kind == "gb":
        return GBClassifier(seed=seed)
    if kind == "partitionwise":
        return PartitionWise(n_groups=n_groups, seed=seed)
    raise ValueError(kind)


# ---------------- Algorithm 1 ----------------
def algorithm1(X, Z, t, losses=("L1", "L2", "KL"), k=5, clf_kind="gb", seed=0,
               cross_fit=True, n_groups=10):
    """Compute \\hat{l-ERT} exactly as in Algorithm 1 (k-fold cross-fitting).

    cross_fit=False is the deliberately broken variant (train and evaluate on the same points).
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    m = len(Z)
    out = {n: 0.0 for n in losses}
    if not cross_fit:
        clf = make_classifier(clf_kind, seed=seed, n_groups=n_groups).fit(X, Z)
        pred = clf.predict_proba1(X)
        return {n: float(np.mean(ert_terms(n, pred, Z, t))) for n in losses}
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for j, (tr, va) in enumerate(kf.split(X)):
        clf = make_classifier(clf_kind, seed=seed + j, n_groups=n_groups).fit(X[tr], Z[tr])
        pred = clf.predict_proba1(X[va])
        for n in losses:
            out[n] += (len(va) / m) * float(np.mean(ert_terms(n, pred, Z[va], t)))
    return out


# ---------------- baseline metrics ----------------
def covgap(X, Z, t, n_groups=10, seed=0, weighted=False):
    km = KMeans(n_clusters=n_groups, n_init=3, random_state=seed).fit(X)
    lab = km.labels_
    gaps, ws = [], []
    for g in range(n_groups):
        m = lab == g
        if m.sum() == 0:
            continue
        gaps.append(abs(Z[m].mean() - t))
        ws.append(m.sum() / len(Z))
    gaps = np.array(gaps)
    ws = np.array(ws)
    return float((ws * gaps).sum()) if weighted else float(gaps.mean())


def wsc(X, Z, t, n_dir=200, delta=0.2, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    best = 1.0
    n = len(Z)
    for _ in range(n_dir):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        proj = X @ v
        order = np.argsort(proj)
        zs = Z[order]
        cum = np.concatenate([[0.0], np.cumsum(zs)])
        need = int(np.ceil(delta * n))
        for i in range(0, n - need + 1, max(1, n // 100)):
            for j in range(i + need, n + 1, max(1, n // 100)):
                cov = (cum[j] - cum[i]) / (j - i)
                if cov < best:
                    best = cov
    return float(best)
