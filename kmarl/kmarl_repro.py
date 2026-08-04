"""
Reproduction of claims A3, A6, A4 from
"Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric
Survival Analysis" (Miyagawa & Ebihara, NEC; ICML 2026; arXiv 2605.18798).

================ TRANSCRIBED SPEC (verbatim/near-verbatim from the PDF) ================

NOTATION (Sec. 3)
  X^(0,t) = (X(0),...,X(t)); frame X(s) ~ g(pre-change) for s < nu, ~ f(post-change)
  for s >= nu. nu in Z>=0 U {inf} is the changepoint, independent of observations.
  T = sequence length (random, independent of observations).
  tau(X^(0,T)) in Z>=0 U {inf} is the detection point; tau = inf if no alarm.
  ARL      mu_inf := E[tau | nu = inf]                                    (Eq. 1)
  ADD      M_inf  := E[dtau | dtau >= 0, nu < inf],  dtau := tau - nu     (Eq. 2)
  (T = inf assumed in Eqs. 1-2.)

CONVENTIONAL ESTIMATORS (Sec. 3)
  LB-ARL   mu^(LB)_T := <tau_i>_{i: nu_i = inf, tau_i <= T_i}
  LB-ADD   M^(LB)_T  := <dtau_i>_{i: dtau_i >= 0, nu_i < inf, dtau_i <= dT_i},
           dT_i := T_i - nu_i
  Naive ARL  mu^(NV)_T := <tau_i>_{i: tau_i < nu_i, tau_i <= T_i}
           "has a non-vanishing bias because E[mu^(NV)_T] = E[tau | tau < nu, tau < T]"

KM-ARL (Eq. 3)
  S_ARL(t) := P(tau > t | nu = inf)
  Shat_ARL(t) = prod_{j: t^ARL_j <= t} (1 - d^ARL_j / n^ARL_j)
    t^ARL_1 < ... < t^ARL_{N'} : distinct detection points
    d^ARL_j := |{i in [N] | tau_i = t^ARL_j}|
    n^ARL_j := |{i in [N] | min(tau_i, C^ARL_i) >= t^ARL_j}|
    C^ARL_i := min(nu_i, T_i)                       (censoring time)
  mu^(KM)_T := int_0^a Shat_ARL(t) dt,  a = Tmax := max_i {min(tau_i, C^ARL_i)}

KM-ADD (Eq. 4)
  S_ADD(t) := P(dtau > t | dtau >= 0, nu < inf)
  Shat_ADD from d^ADD_j := |{i | 0 <= dtau_i = t^ADD_j}|,
    n^ADD_j := |{i | min(dtau_i, C^ADD_i) >= t^ADD_j, dtau_i >= 0}|,
    C^ADD_i := dT_i = T_i - nu_i
  M^(KM)_T := int_0^b Shat_ADD(t) dt,  b = dTmax := max_i{min(dtau_i, C^ADD_i)}

THEOREMS UNDER TEST
  Thm 4.1 : -int_0^a t G(t) H(t)^{N-1} dF(t) <= B_FS(mu^KM) <= int_0^a a G H^{N-1} dF
            => finite-sample bias decays exponentially in N.                     [A1]
  Thm 4.3 : for a = T*max := inf{T | CDF(T) = 1},
            B_TR(mu^(LB)_T) <= B_TR(mu^(KM)_T) <= 0                              [A3]
  Thm 4.4 : for b = dT*max,  B_TR(M^(LB)_T) <= B_TR(M^(KM)_T) <= 0               [A3]
  Eq. 11  : setting a = tF := inf{t | F_ARL(t) = 1}, truncation bias vanishes and
            the bound decays exponentially iff tF < tH := inf{t | H_ARL(t) = 1};
            i.e. KM-ARL is asymptotically unbiased if tF < tH, and NOT otherwise
            ("extrapolation" regime).                                            [A4]

SIMULATION SETUP (Sec. 5, 5.1)
  Detector: generalized Shiryaev-Roberts (GSR) with ground-truth statistics.
    R(t) = omega * prod_{s=0}^t L(s) + sum_{k=0}^t prod_{s=k}^t L(s),  omega = 0.
    => recursion R(t) = (R(t-1) + 1) * L(t), R(-1) = omega = 0.
    Alarm when R(t) hits a pre-defined threshold.
  Gaussian process dataset: pre-change mean 0, post-change mean 0.1, variance 0.1
    for both.  => L(t) = exp((0.1*x - 0.005)/0.1) = exp(x - 0.05).
  Changepoint distributions: uniform (Fig. 2) and geometric (Fig. 3).
  Fig. 2: N = 1000 sequences, uniform changepoints.
    (a) T = 1000 const, 10% of sequences contain a changepoint
    (b) T = 1000 const, 90% contain a changepoint
    (c) T ~ U[100, 1000] irregular, 90% contain a changepoint
    (d) T ~ U[30, 300]  irregular, 90% contain a changepoint (extrapolation regime)
  True ARLs/ADDs: "we simulate the ground-truth ARLs and ADDs by generating
    sequences of effectively infinite length" (relative errors <~ 1e-3).

============================== END TRANSCRIBED SPEC ==============================

Ground truth here follows the paper's own definition (Monte-Carlo on effectively
infinite sequences), cross-checked against the exact geometric/analytic identity
ARL = int_0^inf S_ARL(t) dt on uncensored data.

No scipy / sklearn. numpy only.
"""
import numpy as np

MU1, VAR = 0.1, 0.1
LOGL_SHIFT = MU1 / VAR          # coefficient on x in log L
LOGL_CONST = -0.5 * MU1**2 / VAR


def run_gsr(rng, n_seq, nu, T, thresh, max_len):
    """Vectorised GSR on Gaussian streams. Returns tau (inf coded as max_len+BIG).

    nu: array of changepoint indices (np.inf for no change), T: array of lengths.
    Detection point tau = first t with R(t) >= thresh, searched over t < T_i
    (sequence is only T_i long). Returns tau as float array with np.inf for
    "no alarm within the sequence".
    """
    logthr = np.log(thresh)
    logR = np.full(n_seq, -np.inf)          # R(-1) = 0
    tau = np.full(n_seq, np.inf)
    alive = np.ones(n_seq, dtype=bool)
    for t in range(max_len):
        alive &= (t < T)
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        mean = np.where(t >= nu[idx], MU1, 0.0)
        x = rng.normal(mean, np.sqrt(VAR))
        logl = LOGL_SHIFT * x + LOGL_CONST
        # R(t) = (R(t-1) + 1) * L(t)  ->  logR = log1p(exp(logR)) + logl
        lr = logR[idx]
        logR[idx] = np.logaddexp(lr, 0.0) + logl
        hit = idx[logR[idx] >= logthr]
        if hit.size:
            tau[hit] = t
            alive[hit] = False
    return tau


def true_arl_add(rng, thresh, n=4000, max_len=60000):
    """Ground truth by 'effectively infinite length' sequences (paper's method)."""
    inf_nu = np.full(n, np.inf)
    L = np.full(n, max_len)
    tau0 = run_gsr(rng, n, inf_nu, L, thresh, max_len)      # nu = inf -> ARL
    zero_nu = np.zeros(n)
    tau1 = run_gsr(rng, n, zero_nu, L, thresh, max_len)     # nu = 0    -> ADD
    return np.mean(tau0[np.isfinite(tau0)]), np.mean(tau1[np.isfinite(tau1)]), \
           np.mean(np.isfinite(tau0)), np.mean(np.isfinite(tau1))


def km_integral(event, cens, upper=None):
    """Kaplan-Meier area under curve. event: observed event times (float, inf =
    not observed); cens: censoring times. Follows Eqs. 3/4 exactly.
    last-observed time  Y_i = min(event_i, cens_i);  delta_i = event_i <= cens_i.
    """
    y = np.minimum(np.where(np.isfinite(event), event, np.inf), cens)
    d = np.isfinite(event) & (event <= cens)
    if not d.any():
        return 0.0
    a = y.max() if upper is None else upper
    times = np.unique(y[d])
    times = times[times <= a]
    S, prev_t, area = 1.0, 0.0, 0.0
    for tj in times:
        dj = np.count_nonzero(d & (y == tj))
        nj = np.count_nonzero(y >= tj)
        area += S * (tj - prev_t)          # S is constant on [prev_t, tj)
        S *= (1.0 - dj / nj)
        prev_t = tj
    area += S * (a - prev_t)
    return area


def make_dataset(rng, N, thresh, T_lo, T_hi, prevalence, max_len):
    T = rng.integers(T_lo, T_hi + 1, size=N).astype(float) if T_hi > T_lo \
        else np.full(N, float(T_lo))
    has_cp = rng.random(N) < prevalence
    nu = np.full(N, np.inf)
    nu[has_cp] = rng.integers(0, T[has_cp].astype(int))     # uniform changepoints
    tau = run_gsr(rng, N, nu, T, thresh, max_len)
    return tau, nu, T


def estimators(tau, nu, T):
    """LB-ARL, Naive-ARL, KM-ARL, LB-ADD, KM-ADD per Sec. 3 / Sec. 5.1."""
    out = {}
    # ---- ARL ----
    m = np.isinf(nu) & np.isfinite(tau) & (tau <= T)
    out['LB_ARL'] = tau[m].mean() if m.any() else np.nan
    m = (tau < nu) & np.isfinite(tau) & (tau <= T)
    out['NV_ARL'] = tau[m].mean() if m.any() else np.nan
    C = np.minimum(nu, T)                                   # C^ARL_i
    out['KM_ARL'] = km_integral(tau, C)
    # ---- ADD ----
    cp = np.isfinite(nu)
    dtau = np.where(cp & np.isfinite(tau), tau - nu, np.inf)
    dT = np.where(cp, T - nu, np.nan)
    keep = cp & ((~np.isfinite(dtau)) | (dtau >= 0))
    m = keep & np.isfinite(dtau) & (dtau <= dT)
    out['LB_ADD'] = dtau[m].mean() if m.any() else np.nan
    out['KM_ADD'] = km_integral(dtau[keep], dT[keep]) if keep.any() else np.nan
    return out


def fmt(x):
    return "  nan " if not np.isfinite(x) else f"{x:7.1f}"


if __name__ == "__main__":
    THRESH = 50.0          # GSR threshold; true ARL ~ few hundred
    MAXLEN = 60000
    print("=" * 100)
    g = np.random.default_rng(12345)
    tARL, tADD, f0, f1 = true_arl_add(g, THRESH)
    print(f"GROUND TRUTH (GSR thr={THRESH}, effectively-infinite sequences, n=4000)")
    print(f"  true ARL = {tARL:.2f}   true ADD = {tADD:.3f}   "
          f"(alarm rate {f0:.4f}/{f1:.4f})")

    SEEDS = list(range(10))
    N = 1000

    # ---------- A3 / A6 : bias vs censoring level, fixed length ----------
    for label, (T_lo, T_hi) in [("T=1000 const", (1000, 1000)),
                                ("T~U[100,1000] irregular", (100, 1000)),
                                ("T~U[500,1000] irregular", (500, 1000))]:
        print("\n" + "=" * 100)
        print(f"SETUP: {label}, N={N}, uniform changepoints, 10 seeds")
        print(f"{'prev':>5} | {'KM-ARL bias':>22} | {'LB-ARL bias':>22} | "
              f"{'NV-ARL bias':>22} | KM<LB")
        for prev in [0.1, 0.3, 0.5, 0.7, 0.9]:
            rows = []
            for s in SEEDS:
                r = np.random.default_rng(1000 + s)
                tau, nu, T = make_dataset(r, N, THRESH, T_lo, T_hi, prev, MAXLEN)
                e = estimators(tau, nu, T)
                rows.append([e['KM_ARL'] - tARL, e['LB_ARL'] - tARL,
                             e['NV_ARL'] - tARL])
            a = np.array(rows)
            win = np.mean(np.abs(a[:, 0]) < np.abs(a[:, 1]))
            print(f"{prev:>5.1f} | "
                  + " | ".join(f"{a[:,k].mean():+8.1f} +/- {a[:,k].std():6.1f}"
                               f" [{np.nanmin(a[:,k]):+7.1f},{np.nanmax(a[:,k]):+7.1f}]"[:22]
                               for k in range(3))
                  + f" | {win*10:.0f}/10")
            print("      | " + " | ".join(
                f"rel {a[:,k].mean()/tARL:+.3f} sd {a[:,k].std()/tARL:.3f}".ljust(22)
                for k in range(3)))

    # ---------- ADD (Thm 4.4) ----------
    print("\n" + "=" * 100)
    print("ADD: geometric changepoints (Fig. 3 style), N=10000, 10 seeds")
    print(f"{'setup':>26} | {'KM-ADD bias':>20} | {'LB-ADD bias':>20} | KM<LB")
    for name, (T_lo, T_hi, p) in [("T=100, p=0.25", (100, 100, 0.25)),
                                  ("T=100, p=0.001", (100, 100, 0.001)),
                                  ("T~U[10,100], p=0.001", (10, 100, 0.001))]:
        rows = []
        for s in SEEDS:
            r = np.random.default_rng(2000 + s)
            NN = 10000
            T = (r.integers(T_lo, T_hi + 1, NN).astype(float) if T_hi > T_lo
                 else np.full(NN, float(T_lo)))
            nu = np.floor(r.geometric(p, NN) - 1).astype(float)
            nu[nu >= T] = np.inf            # changepoint beyond horizon = no change
            tau = run_gsr(r, NN, nu, T, THRESH, int(T_hi))
            e = estimators(tau, nu, T)
            rows.append([e['KM_ADD'] - tADD, e['LB_ADD'] - tADD])
        a = np.array(rows)
        win = np.mean(np.abs(a[:, 0]) < np.abs(a[:, 1]))
        print(f"{name:>26} | " + " | ".join(
            f"{a[:,k].mean():+7.2f} +/- {a[:,k].std():5.2f}".ljust(20) for k in range(2))
            + f" | {win*10:.0f}/10")

    # ---------- A4 : asymptotic unbiasedness, support condition on/off ----------
    print("\n" + "=" * 100)
    print("A4: KM-ARL bias vs N.  Condition tF < tH (support of detection points")
    print("    within censoring boundary) SATISFIED vs VIOLATED.  prevalence 0.5")
    print(f"{'N':>7} | {'SATISFIED T=5000':>26} | {'VIOLATED T=100':>26}")
    for NN in [50, 200, 1000, 5000, 20000]:
        sat, vio = [], []
        for s in SEEDS:
            r = np.random.default_rng(3000 + s)
            tau, nu, T = make_dataset(r, NN, THRESH, 5000, 5000, 0.5, MAXLEN)
            sat.append(estimators(tau, nu, T)['KM_ARL'] - tARL)
            r = np.random.default_rng(4000 + s)
            tau, nu, T = make_dataset(r, NN, THRESH, 100, 100, 0.5, MAXLEN)
            vio.append(estimators(tau, nu, T)['KM_ARL'] - tARL)
        sat, vio = np.array(sat), np.array(vio)
        print(f"{NN:>7} | {sat.mean():+8.2f} +/- {sat.std():6.2f} (rel {sat.mean()/tARL:+.3f})"
              f" | {vio.mean():+8.2f} +/- {vio.std():6.2f} (rel {vio.mean()/tARL:+.3f})")
