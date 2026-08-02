"""common.py — shared primitives for reproducing 418BWmKIzX.

All runs are CPU-only, numpy/scipy/sympy only. No external data.

Key invariant (used for EXACT, cheap error measurement):
  For target normal w* (unit) and hypothesis w (unit) under constant-eta RCN on the
  unit sphere, the true prediction error is
      err = eta + (1-2 eta) * P_x(sign(w.x) != sign(w*.x)),   x~Uniform(S^{d-1}),
  and P_x(disagree) = angle(w, w*) / pi.
  Hence excess error beyond the Massart floor eta equals (1-2 eta) * angle/pi.
  In the realizable case (eta=0) the error IS the disagreement probability.
"""
import numpy as np


def proj_unit(w):
    n = np.linalg.norm(w)
    if n <= 0:
        return w
    return w / n


def random_unit(d, rng):
    return proj_unit(rng.standard_normal(d))


def make_drift_targets(d, T, Delta, rng, base_step=None):
    """Random walk on the unit sphere with controlled per-step TV<=Delta.

    The paper requires d_TV(D^(t),D^(t+1))<=Delta. For uniform-sphere halfspaces with
    margin gamma, disagreement prob = angle/pi, so per-step angle must be <= O(Delta).
    A diffusive random walk of step `s` accumulates angle ~ sqrt(W)*s over an epoch of
    length W=Theta(Delta^{-2/3}); choosing s = Theta(Delta^{2/3}) makes the cumulative
    angle ~ sqrt(Delta^{-2/3})*Delta^{2/3} = Theta(Delta^{1/3}), exactly the drift term
    in Theorem 1.1's bound. (Per-step TV ~ s/pi << Delta.)
    """
    if base_step is None:
        base_step = 0.3 * (max(Delta, 1e-6) ** (2.0 / 3.0))
    ws = np.empty((T, d))
    ws[0] = np.eye(d)[0]  # e1
    for t in range(1, T):
        delta = rng.standard_normal(d) * base_step
        ws[t] = proj_unit(ws[t - 1] + delta)
    return ws


def angle_between(a, b):
    """Angle (radians) between two unit vectors."""
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(c)


def excess_error(w_hat, w_star, eta):
    """Exact excess error beyond the Massart floor eta for unit vectors on the sphere."""
    ang = angle_between(w_hat, w_star)
    disc = ang / np.pi
    return (1.0 - 2.0 * eta) * disc


def sample_example(w_star, eta, rng, realizable=False):
    """One labeled example from D realized by (w_star, eta). x~Uniform(S^{d-1})."""
    d = w_star.shape[0]
    x = random_unit(d, rng)
    wx = float(np.dot(w_star, x))
    true = 1.0 if wx >= 0 else -1.0
    if realizable:
        y = true
    else:
        if rng.random() < eta:
            y = -true
        else:
            y = true
    return x, y


def drift_perceptron(S, eta, gamma, mu, realizable=False, massart_grad=True,
                     init=None):
    """Algorithm 2 (DriftPerceptron). S: (2m, d+1) array of (x,y).
    Returns (best_w, best_empirical_err, traj) where traj is the list of
    (w_i, empirical_err_i) for i=1..m.
    massart_grad=False reproduces the naive-perceptron MUTATION (drops (1-2eta))."""
    S = np.asarray(S, dtype=float)
    X = S[:, :-1]
    Y = S[:, -1]
    m = len(S) // 2
    X1, Y1 = X[:m], Y[:m]
    X2, Y2 = X[m:], Y[m:]
    d = X.shape[1]
    w = np.zeros(d)
    if init is None:
        w[0] = 1.0
    else:
        w = np.array(init, dtype=float)
    best_w = w.copy()
    best_err = np.inf
    traj = []
    for i in range(m):
        x, y = X1[i], Y1[i]
        wx = float(np.dot(w, x))
        denom = max(abs(wx), gamma)
        if realizable:
            g = (np.sign(wx) - y) * x
        else:
            if massart_grad:
                g = ((1.0 - 2.0 * eta) * np.sign(wx) - y) * x / denom
            else:
                g = (np.sign(wx) - y) * x  # naive perceptron (mutation)
        w = proj_unit(w - mu * g)
        pred = np.sign(X2 @ w)
        pred[pred == 0] = 1.0
        err = float(np.mean(pred != Y2))
        traj.append((w.copy(), err))
        if err < best_err:
            best_err = err
            best_w = w.copy()
    return best_w, best_err, traj


def run_drifted_massart(d, Delta, gamma, eta, n_epochs, rng,
                        realizable=False, massart_grad=True,
                        epoch_len=None, no_epoch=False, base_step=None,
                        no_drift=False):
    """Algorithm 1 (DriftedMassart). Returns dict with per-epoch excess errors
    (measured exactly via the angle formula) and the final hypothesis.

    epoch_len: override W. no_epoch=True sets W = total steps (single epoch, never
    resets) — this is mutation M2 (disabling drift-forgetting).
    no_drift=True (mutation M1): static target, fixed epoch length.
    """
    if no_drift:
        # static target, fixed (modest) epoch length; measure collapse of excess.
        Delta_eff = 0.016
        if base_step is None:
            base_step = 0.0
    else:
        Delta_eff = Delta
    # epoch length following the paper: W = 2 floor(Delta^{-2/3} log(1/Delta))
    if epoch_len is None:
        if no_epoch:
            epoch_len = None
        else:
            Wraw = 2.0 * (Delta_eff ** (-2.0 / 3.0)) * max(1.0, np.log(1.0 / Delta_eff))
            epoch_len = max(2, int(round(Wraw)))
            if epoch_len % 2 == 1:
                epoch_len += 1
    if no_drift:
        epoch_len = max(2, int(round(
            2.0 * (0.016 ** (-2.0 / 3.0)) * max(1.0, np.log(1.0 / 0.016)))))
    T_total = n_epochs * epoch_len if (epoch_len and not no_epoch) else (
        epoch_len if epoch_len else 200)
    if no_epoch:
        epoch_len = T_total = max(50, int(round(
            2.0 * (Delta ** (-2.0 / 3.0)) * max(1.0, np.log(1.0 / Delta)) * n_epochs)))
    targets = make_drift_targets(d, T_total, Delta, rng, base_step=base_step)
    # mu = gamma / sqrt(W) per Algorithm 1; T (inner) = W/2.
    mu = gamma / np.sqrt(epoch_len)
    w_hat = np.zeros(d)
    w_hat[0] = 1.0  # h^0(x) = 1
    epoch_excess = []
    all_excess = []
    step = 0
    n_eps = n_epochs if not no_epoch else 1
    for ep in range(n_eps):
        S = []
        start = ep * epoch_len if not no_epoch else 0
        end = start + epoch_len
        for t in range(start, end):
            x, y = sample_example(targets[t], eta, rng, realizable=realizable)
            S.append(np.concatenate([x, [y]]))
        S = np.array(S, dtype=float)
        if realizable:
            best_w, _, _ = drift_perceptron(S, eta, gamma, mu, realizable=True)
        else:
            best_w, _, _ = drift_perceptron(S, eta, gamma, mu, realizable=False,
                                            massart_grad=massart_grad)
        w_hat = best_w
        # measure exact excess error of w_hat against the LAST target in this epoch
        w_star = targets[end - 1]
        ex = excess_error(w_hat, w_star, 0.0 if realizable else eta)
        epoch_excess.append(float(ex))
        all_excess.append(float(ex))
    return {
        "epoch_len": int(epoch_len),
        "n_epochs": int(n_eps),
        "epoch_excess": epoch_excess,
        "final_excess": float(epoch_excess[-1]),
        "mean_epoch_excess": float(np.mean(epoch_excess)),
        "w_hat": w_hat,
    }


def logfit(xs, ys):
    """Least-squares slope of log(y) vs log(x). Returns (slope, intercept, r2)."""
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = coef
    pred = A @ coef
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)
