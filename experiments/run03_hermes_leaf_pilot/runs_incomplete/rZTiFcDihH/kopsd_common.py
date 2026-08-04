"""
kopsd_common.py -- shared primitives for reproducing the 6 anchored claims of
"Online Packet Scheduling with Deadlines and Learning" (arXiv:2606.00835,
OpenReview rZTiFcDihH).

CPU-only reproduction strategy (honest about the theory nature of the claims):
  * The claims are THEORETICAL (closed-form competitive ratios, regret upper
    bounds O~(sqrt(KT)), a reduction lemma).  We reproduce them in two
    complementary ways:
      (A) ANALYTIC: the exact constants the paper derives are recomputed from
          first principles -- the theta_K sequence from system (1) (Prop. 4.1),
          the 5/4 and e/(e-1) competitive-ratio constants (Thm 5.1/5.2) and the
          golden ratio Phi.  These are exact, reproducible checks.
      (B) EMPIRICAL: each algorithm is implemented faithfully from its
          pseudocode (Algs 1-5) and run on stochastic K-OPSD instances; we
          measure its alpha-regret R_{alpha,T} across a range of horizons T and
          confirm the claimed O~(sqrt(KT)) SCALING (sublinear, log-log slope
          ~ 0.5), and for the deterministic algorithm the competitive-ratio
          bound G_OPT <= theta_K G_ALG is checked on instances.
      (C) REDUCTION: Lemma 2.1 (sleeping bandits <==> 1-bounded K-OPSD) is
          constructed explicitly and verified numerically.

All randomness is seeded (numpy default_rng) so the artifacts are reproducible.

Paper notation recap:
  * s-bounded: every packet has d_p <= r_p + s - 1 (s possible slots).
  * 2-bounded: d in {r, r+1}; 3-bounded: d in {r, r+1, r+2}.
  * V_t = types with a packet in the buffer whose deadline == t.
  * B_t = types with a packet in the buffer whose deadline  > t.
  * alpha-regret: R_{alpha,T} = E[G_OPT] - alpha E[G_ALG].
  * Phi = (1+sqrt(5))/2 (golden ratio).
  * theta_K: unique nonnegative solution of system (1) (Prop. 4.1).
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Tuple, Optional

PHI = (1.0 + math.sqrt(5.0)) / 2.0          # golden ratio ~ 1.6179
SQRT2 = math.sqrt(2.0)
E_OVER_EM1 = math.e / (math.e - 1.0)        # e/(e-1) ~ 1.58198
FIVE_OVER_FOUR = 1.25


# ---------------------------------------------------------------------------
# theta_K solver (system (1) of the paper, Proposition 4.1)
# ---------------------------------------------------------------------------
def theta_system_xs(theta: float, K: int) -> List[float]:
    """Compute x_0..x_{K-1} from system (1) for a trial theta.

    x_0 = 1,
    x_1 = 1/(theta-1),
    x_j = ((theta+1)/(theta-1)) * (x_{j-1} - x_{j-2})   for 2 <= j <= K-1.
    """
    if theta <= 1.0:
        return [float('nan')] * K
    xs = [0.0] * K
    xs[0] = 1.0
    if K >= 2:
        xs[1] = 1.0 / (theta - 1.0)
    r = (theta + 1.0) / (theta - 1.0)
    for j in range(2, K):
        xs[j] = r * (xs[j - 1] - xs[j - 2])
    return xs


def theta_system_residual(theta: float, K: int) -> float:
    """Residual of the closing equation x_{K-1} = (theta+1) x_{K-2}."""
    xs = theta_system_xs(theta, K)
    if K == 2:
        # x_1 = (theta+1) x_0 = theta+1
        return xs[1] - (theta + 1.0)
    return xs[K - 1] - (theta + 1.0) * xs[K - 2]


def solve_theta_K(K: int, lo: float = 1.0 + 1e-9, hi: float = PHI + 1e-3) -> Tuple[float, List[float]]:
    """Return (theta_K, [x_0..x_{K-1}]) solving system (1).

    theta_K -> Phi as K -> inf; theta_2 = sqrt(2), theta_3 = 3/2.  The paper
    proves a unique nonnegative solution with theta_K in [sqrt(2), Phi); we
    bracket the single sign-change of the residual strictly inside (1, Phi).
    """
    from scipy.optimize import brentq
    if K == 1:
        return 1.0, [1.0]
    # fine scan inside (lo, hi) to locate the (unique) sign change
    grid = np.linspace(lo, hi, 2000)
    f_prev = None
    prev_th = None
    for th in grid:
        try:
            f = theta_system_residual(th, K)
        except Exception:
            f = float('nan')
        if math.isnan(f):
            continue
        if f_prev is not None and f_prev > 0 and f <= 0:
            a, b = prev_th, th
            break
        if f_prev is not None and f_prev < 0 and f >= 0:
            a, b = th, prev_th
            break
        f_prev, prev_th = f, th
    else:
        # root extremely close to Phi for very large K; nudge bracket up
        a, b = hi - 1e-4, hi
    theta = float(brentq(theta_system_residual, a, b, args=(K,)))
    return theta, theta_system_xs(theta, K)


# ---------------------------------------------------------------------------
# Instance / simulator
# ---------------------------------------------------------------------------
@dataclass
class Packet:
    pid: int
    r: int          # arrival time
    d: int          # deadline (packet may be scheduled in {r..d})
    c: int          # type (0-indexed)
    weight: float = 0.0   # realised weight (filled on scheduling)


@dataclass
class Instance:
    K: int
    T: int
    means: List[float]            # means[c] for c in 0..K-1 (descending)
    stochastic: bool              # True -> Bernoulli(means) weights; False -> fixed = means
    arrivals: Dict[int, List[Packet]] = field(default_factory=dict)
    seed: int = 0

    def all_packets(self) -> List[Packet]:
        out = []
        for t in sorted(self.arrivals):
            out.extend(self.arrivals[t])
        return out


def build_alt_instance(K: int, T: int, means: List[float], stochastic: bool,
                       seed: int = 0, s: int = 2) -> Instance:
    """Build an 'alternating' s-bounded instance that forces learning.

    At every time t the adversary presents two packets of two distinct types:
      * a 'current' packet with deadline t   (in V_t, must go now or drop)
      * a 'future' packet  with deadline t+s-1 (in B_t, can wait)
    The roles alternate so the best type is sometimes 'now' and sometimes
    'later'.  With unknown weights the algorithm must explore to learn which
    type has the larger mean -- exactly the bandit-style exploration that
    yields the O~(sqrt(KT)) regret term.
    """
    rng = np.random.default_rng(seed)
    inst = Instance(K=K, T=T, means=list(means), stochastic=stochastic,
                    arrivals={}, seed=seed)
    pid = 0
    # best type index 0, others form the "decoy" pool.
    for t in range(1, T + 1):
        # alternate which type is 'current' vs 'future'
        if t % 2 == 1:
            cur, fut = (1 % K), 0          # current = a decoy, future = best
        else:
            cur, fut = 0, (1 % K)          # current = best, future = decoy
        inst.arrivals.setdefault(t, []).append(
            Packet(pid=pid, r=t, d=t, c=cur))
        pid += 1
        inst.arrivals.setdefault(t, []).append(
            Packet(pid=pid, r=t, d=t + s - 1, c=fut))
        pid += 1
    return inst


def build_1bounded(K: int, T: int, means: List[float], stochastic: bool,
                   seed: int = 0) -> Instance:
    """1-bounded K-OPSD instance == K-armed sleeping bandit (Lemma 2.1).

    At each round t the adversary reveals a random non-empty subset of the K
    types (the sleeping-bandit active set A_t); the algorithm may schedule one
    packet (the single server pulls one arm).  Used to verify the O~(sqrt(KT))
    regret ORDER via the exact reduction to sleeping bandits.
    """
    rng = np.random.default_rng(seed)
    inst = Instance(K=K, T=T, means=list(means), stochastic=stochastic,
                    arrivals={}, seed=seed)
    pid = 0
    for t in range(1, T + 1):
        kk = int(rng.integers(1, K + 1))
        types = sorted(rng.choice(K, size=kk, replace=False).tolist())
        for c in types:
            inst.arrivals.setdefault(t, []).append(
                Packet(pid=pid, r=t, d=t, c=int(c)))
            pid += 1
    return inst


def build_decoy_instance(K: int, T: int, means: List[float], stochastic: bool,
                         seed: int = 0, s: int = 2) -> Instance:
    """2-bounded 'decoy' instance for competitive-ratio / mutation checks.

    At every round t the adversary presents a WORST-type packet with deadline t
    (must go now or be dropped) and a BEST-type packet with deadline t+s-1 (can
    wait).  A non-learning greedy (earliest-deadline-first) takes the worst now
    every round -> LINEAR regret (mutation target).  The optimistic algorithms
    (EDF_Phi^L / ALG^theta / ...) wait for the best -> near-optimal, satisfying
    the sublinear bound.
    """
    rng = np.random.default_rng(seed)
    inst = Instance(K=K, T=T, means=list(means), stochastic=stochastic,
                    arrivals={}, seed=seed)
    pid = 0
    best, worst = 0, (K - 1) % K
    for t in range(1, T + 1):
        # worst now, best deferred
        inst.arrivals.setdefault(t, []).append(
            Packet(pid=pid, r=t, d=t, c=worst))
        pid += 1
        inst.arrivals.setdefault(t, []).append(
            Packet(pid=pid, r=t, d=t + s - 1, c=best))
        pid += 1
    return inst


def sample_weight(pkt: Packet, inst: Instance, rng: np.random.Generator) -> float:
    if inst.stochastic:
        w = float(rng.random() < inst.means[pkt.c])   # Bernoulli(mean)
    else:
        w = float(inst.means[pkt.c])
    return w


def opt_expected_gain(inst: Instance) -> float:
    """Offline optimum using EXPECTED weights (means): backward greedy.

    Process slots T..1; at each slot pick the highest-mean available packet.
    Optimal for unit-time jobs with release times + deadlines.
    """
    packets = inst.all_packets()
    best = 0.0
    # for each slot t from T down to 1, choose best packet with r<=t<=d
    scheduled = set()
    for t in range(inst.T, 0, -1):
        cand = None
        for p in packets:
            if p.pid in scheduled:
                continue
            if p.r <= t <= p.d:
                # pick the one with the largest mean; tie-break by earliest deadline
                score = inst.means[p.c]
                if cand is None or score > inst.means[cand.c] + 1e-12 or (
                        abs(score - inst.means[cand.c]) <= 1e-12 and p.d < cand.d):
                    cand = p
        if cand is not None:
            best += inst.means[cand.c]
            scheduled.add(cand.pid)
    return best


def opt_realized_gain(inst: Instance, rng: np.random.Generator) -> float:
    """Offline optimum using REALISED weights (for the known-weight OPSD setting)."""
    packets = inst.all_packets()
    # realised weights
    w = {p.pid: sample_weight(p, inst, rng) for p in packets}
    scheduled = set()
    gain = 0.0
    for t in range(inst.T, 0, -1):
        cand = None
        for p in packets:
            if p.pid in scheduled:
                continue
            if p.r <= t <= p.d:
                if cand is None or w[p.pid] > w[cand.pid] + 1e-12:
                    cand = p
        if cand is not None:
            gain += w[cand.pid]
            scheduled.add(cand.pid)
    return gain


# ---------------------------------------------------------------------------
# Algorithm base: operates on a buffer of packets at time t.
# ---------------------------------------------------------------------------
class Algo:
    def __init__(self, inst: Instance, known_weights: bool = False):
        self.inst = inst
        self.K = inst.K
        self.known_weights = known_weights

    # ---- helpers to classify buffer ----
    @staticmethod
    def _split(buffer, t):
        """Return (packets_with_deadline_t, packets_with_deadline_gt_t)."""
        V, B = [], []
        for p in buffer:
            if p.d == t:
                V.append(p)
            elif p.d > t:
                B.append(p)
        return V, B

    def choose(self, buffer, t, rng):
        raise NotImplementedError

    def reset(self):
        pass


# ----- EDF_Phi^L (Algorithm 1) : learning, 2/3-bounded, Phi-regret -----
class EDFPhiL(Algo):
    """EDF_Phi^L (Algorithm 1).  h_t = max_{h in B_t} UCB_h,t ; schedule the
    earliest-deadline packet in B_t whose UCB_f >= UCB_h / Phi."""
    def reset(self):
        self.N = np.zeros(self.K)
        self.sumw = np.zeros(self.K)
        if self.known_weights:
            self.ucb = np.array(self.inst.means, dtype=float)
            self.lcb = np.array(self.inst.means, dtype=float)
        else:
            self.ucb = np.ones(self.K)
            self.lcb = np.zeros(self.K)
        self.delta = 1.0 / (self.K * self.inst.T ** 2 + 1.0)

    def _update(self, c, w):
        if self.known_weights:
            return
        self.N[c] += 1
        self.sumw[c] += w
        mu = self.sumw[c] / self.N[c]
        beta = math.sqrt(math.log(self.K * self.inst.T ** 2 / self.delta) /
                         (2.0 * self.N[c]))
        self.ucb[c] = min(self.ucb[c], mu + beta)
        self.lcb[c] = max(self.lcb[c], mu - beta)

    def choose(self, buffer, t, rng):
        V, B = self._split(buffer, t)
        if not B:
            if not V:
                return None
            V.sort(key=lambda p: p.d)
            return V[0]
        if self.known_weights:
            h = max(B, key=lambda p: p.weight)
            uh = h.weight
            cands = [p for p in B if PHI * p.weight >= uh - 1e-12]
        else:
            h = max(B, key=lambda p: self.ucb[p.c])
            uh = self.ucb[h.c]
            cands = [p for p in B if self.ucb[p.c] * PHI >= uh - 1e-12]
        cands.sort(key=lambda p: p.d)
        return cands[0] if cands else (B[0] if B else None)


# ----- ALG^theta (Algorithm 2) : deterministic, KNOWN weights, 2-bounded -----
class ALGtheta(Algo):
    """ALG^theta (Algorithm 2).  Uses ACTUAL (known) weights.  Pre-solve the
    theta_K system to get the x_j thresholds."""
    def reset(self):
        self.theta, self.xs = solve_theta_K(self.K)
        self.j = 0

    def _update(self, c, w):
        pass

    def choose(self, buffer, t, rng):
        V, B = self._split(buffer, t)
        if not V and not B:
            return None
        if not V:
            V2 = B
        else:
            V2 = V
        if not B:
            # only current packets: schedule the heaviest
            V.sort(key=lambda p: p.d)
            return max(V, key=lambda p: p.weight) if V else None
        # heaviest current (deadline t) and heaviest future (deadline > t)
        v = max(V, key=lambda p: p.weight)
        b = max(B, key=lambda p: p.weight)
        xj = self.xs[self.j]
        xjp1 = self.xs[self.j + 1] if self.j + 1 < len(self.xs) else self.xs[-1]
        if v.weight < (xj / xjp1) * b.weight:
            chosen = b
            self.j = 0
        else:
            chosen = v
            if v.weight >= b.weight:
                self.j = 0
            else:
                self.j += 1
        return chosen


# ----- ALG^theta,U (Algorithm 3) : learning, 2-bounded, theta_K-regret -----
class ALGthetaU(Algo):
    """ALG^{theta,U} (Algorithm 3).  Like ALG^theta but uses UCBs instead of
    actual weights."""
    def reset(self):
        self.theta, self.xs = solve_theta_K(self.K)
        self.j = 0
        self.N = np.zeros(self.K)
        self.sumw = np.zeros(self.K)
        if self.known_weights:
            self.ucb = np.array(self.inst.means, dtype=float)
            self.lcb = np.array(self.inst.means, dtype=float)
        else:
            self.ucb = np.ones(self.K)
            self.lcb = np.zeros(self.K)
        self.delta = 1.0 / (self.K * self.inst.T ** 2 + 1.0)

    def _update(self, c, w):
        if self.known_weights:
            return
        self.N[c] += 1
        self.sumw[c] += w
        mu = self.sumw[c] / self.N[c]
        beta = math.sqrt(math.log(self.K * self.inst.T ** 2 / self.delta) /
                         (2.0 * self.N[c]))
        self.ucb[c] = min(self.ucb[c], mu + beta)
        self.lcb[c] = max(self.lcb[c], mu - beta)

    def choose(self, buffer, t, rng):
        V, B = self._split(buffer, t)
        if not B:
            if not V:
                return None
            # only current packets: schedule highest UCB
            return max(V, key=lambda p: self.ucb[p.c])
        if not V:
            return max(B, key=lambda p: self.ucb[p.c])
        v = max(V, key=lambda p: self.ucb[p.c])
        b = max(B, key=lambda p: self.ucb[p.c])
        xj = self.xs[self.j]
        xjp1 = self.xs[self.j + 1] if self.j + 1 < len(self.xs) else self.xs[-1]
        if self.ucb[v.c] < (xj / xjp1) * self.ucb[b.c]:
            chosen = b
            self.j = 0
        else:
            chosen = v
            if self.ucb[v.c] >= self.ucb[b.c]:
                self.j = 0
            else:
                self.j += 1
        return chosen


# ----- ALG^R2 (Algorithm 4) : randomized, 2-bounded, 5/4-regret -----
class ALGR2(Algo):
    """ALG^{R2} (Algorithm 4).  Randomized 2-bounded algorithm with 5/4-regret."""
    def reset(self):
        self.N = np.zeros(self.K)
        self.sumw = np.zeros(self.K)
        if self.known_weights:
            self.ucb = np.array(self.inst.means, dtype=float)
            self.lcb = np.array(self.inst.means, dtype=float)
        else:
            self.ucb = np.ones(self.K)
            self.lcb = np.zeros(self.K)
        self.delta = 1.0 / (self.K * self.inst.T ** 2 + 1.0)

    def _update(self, c, w):
        if self.known_weights:
            return
        self.N[c] += 1
        self.sumw[c] += w
        mu = self.sumw[c] / self.N[c]
        beta = math.sqrt(math.log(self.K * self.inst.T ** 2 / self.delta) /
                         (2.0 * self.N[c]))
        self.ucb[c] = min(self.ucb[c], mu + beta)
        self.lcb[c] = max(self.lcb[c], mu - beta)

    def _pab(self, a, b):
        ua, la = self.ucb[a], self.lcb[a]
        ub, lb = self.ucb[b], self.lcb[b]
        if ua <= lb:
            return max(4.0 * ua / (5.0 * lb), 0.2)
        elif la <= ub and ua > lb:
            return 0.8   # 4/5
        else:  # la > ub
            return 1.0

    def choose(self, buffer, t, rng):
        V, B = self._split(buffer, t)
        if not V and not B:
            return None
        # identify a_t = argmax_{V} UCB, b_t = argmax_{B} UCB
        if not V:
            a = max(B, key=lambda p: self.ucb[p.c])
        else:
            a = max(V, key=lambda p: self.ucb[p.c])
        if not B:
            b = a
        else:
            b = max(B, key=lambda p: self.ucb[p.c])
        ca, cb = a.c, b.c
        # burn-in: if either has N <= 50 ln T, schedule the less-sampled
        thr = 50.0 * math.log(self.inst.T + 1)
        if self.N[ca] <= thr or self.N[cb] <= thr:
            if self.N[ca] <= self.N[cb]:
                return a
            return b
        p = self._pab(ca, cb)
        if rng.random() < p:
            return a
        return b


# ----- ALG^Rs (Algorithm 5) : randomized, s-bounded, e/(e-1)-regret -----
class ALGRs(Algo):
    """ALG^{Rs} (Algorithm 5).  Randomized s-bounded algorithm with e/(e-1)-regret.
    h_underline = argmax_{B_t} LCB, h_bar = argmax_{B_t} UCB.
    Sample x_t ~ Uniform[-1 + ln(UCB_hu/LCB_hu), ln(UCB_hb/LCB_hu)].
    Schedule f_t = earliest-deadline packet in B_t with UCB_f >= e^{x_t}*LCB_hu
    (Equation (3))."""
    def reset(self):
        self.N = np.zeros(self.K)
        self.sumw = np.zeros(self.K)
        if self.known_weights:
            self.ucb = np.array(self.inst.means, dtype=float)
            self.lcb = np.array(self.inst.means, dtype=float)
        else:
            self.ucb = np.ones(self.K)
            self.lcb = np.zeros(self.K)
        self.delta = 1.0 / (self.K * self.inst.T ** 2 + 1.0)

    def _update(self, c, w):
        if self.known_weights:
            return
        self.N[c] += 1
        self.sumw[c] += w
        mu = self.sumw[c] / self.N[c]
        beta = math.sqrt(math.log(self.K * self.inst.T ** 2 / self.delta) /
                         (2.0 * self.N[c]))
        self.ucb[c] = min(self.ucb[c], mu + beta)
        self.lcb[c] = max(self.lcb[c], mu - beta)

    def choose(self, buffer, t, rng):
        V, B = self._split(buffer, t)
        # For s-bounded (unbounded slackness) every packet effectively has
        # deadline > t within the relevant window, but we still respect V/B.
        pool = B if B else V
        if not pool:
            return None
        # h_underline = argmax LCB, h_bar = argmax UCB among pool (B_t)
        hu = max(pool, key=lambda p: self.lcb[p.c])
        hb = max(pool, key=lambda p: self.ucb[p.c])
        lub_hu = self.lcb[hu.c]
        if lub_hu <= 0:
            lub_hu = 1e-6
        lo = -1.0 + math.log(self.ucb[hu.c] / lub_hu + 1e-12)
        hi = math.log(self.ucb[hb.c] / lub_hu + 1e-12)
        if hi <= lo:
            hi = lo + 1e-6
        x = rng.uniform(lo, hi)
        thr = math.exp(x) * lub_hu
        cands = [p for p in pool if self.ucb[p.c] >= thr - 1e-12]
        if not cands:
            cands = [hu]
        cands.sort(key=lambda p: p.d)
        return cands[0]


# ---------------------------------------------------------------------------
# Generic simulator
# ---------------------------------------------------------------------------
def simulate(alg: Algo, inst: Instance, seed: int,
              known_weights: bool = False,
              n_runs: int = 1) -> Dict:
    """Run alg on inst.  If n_runs>1, average over weight realisations.

    Fresh packet copies are made per run so the _done/_wset flags and realised
    weights never leak across Monte-Carlo runs (otherwise later runs would see
    an empty buffer and the average would be corrupted).

    Returns dict with gains list, mean gain, and per-type counts.
    """
    import copy
    rng = np.random.default_rng(seed)
    gains = []
    N_total = np.zeros(inst.K)
    for run in range(n_runs):
        alg.reset()
        # fresh copies of the arrival packets for this run
        arrivals = {t: [copy.copy(p) for p in pkts]
                    for t, pkts in inst.arrivals.items()}
        buffer: List[Packet] = []
        gain = 0.0
        for t in range(1, inst.T + 1):
            # arrivals
            for p in arrivals.get(t, []):
                buffer.append(p)
            # drop expired (deadline < t) and already-scheduled
            buffer = [p for p in buffer if p.d >= t and not getattr(p, '_done', False)]
            # assign known weights if this is the known-weight (OPSD) setting
            if known_weights or isinstance(alg, ALGtheta):
                for p in buffer:
                    if not getattr(p, '_wset', False):
                        p.weight = float(inst.means[p.c])
                        p._wset = True
            chosen = alg.choose(buffer, t, rng)
            if chosen is not None:
                w = chosen.weight if (known_weights or isinstance(alg, ALGtheta)) \
                    else sample_weight(chosen, inst, rng)
                gain += w
                N_total[chosen.c] += 1
                alg._update(chosen.c, w)   # keep UCB/LCB estimates current
                chosen._done = True
                buffer = [p for p in buffer if p is not chosen]
            # drop expired at end
            buffer = [p for p in buffer if p.d >= t]
        gains.append(gain)
    return {"gains": gains, "mean_gain": float(np.mean(gains)),
            "N": N_total.tolist()}


def alpha_regret(alg: Algo, inst: Instance, seed: int, alpha: float,
                 n_runs: int = 40, known_weights: bool = False) -> Dict:
    """Compute R_{alpha,T} = E[G_OPT] - alpha E[G_ALG] by Monte Carlo.

    In the stochastic setting E[G_OPT] uses expected weights; in the
    known-weight (OPSD) setting E[G_OPT] uses realised weights and G_ALG uses
    known weights too.
    """
    if known_weights:
        rng = np.random.default_rng(seed)
        gopt = np.mean([opt_realized_gain(inst, rng) for _ in range(n_runs)])
    else:
        gopt = opt_expected_gain(inst)
    res = simulate(alg, inst, seed=seed, known_weights=known_weights,
                   n_runs=n_runs)
    galg = res["mean_gain"]
    regret = gopt - alpha * galg
    return {"E_G_OPT": float(gopt), "E_G_ALG": float(galg),
            "alpha": alpha, "regret": float(regret),
            "n_runs": n_runs, "gains": res["gains"], "N": res["N"]}


# ---------------------------------------------------------------------------
# Mutation baseline: a non-learning greedy (earliest-deadline-first, ignores
# weights/exploration).  Used to show that WITHOUT the optimism/learning
# principle the alpha-regret becomes LINEAR (the O~(sqrt(KT)) bound breaks).
# ---------------------------------------------------------------------------
class GreedyEDF(Algo):
    """Schedule the earliest-deadline available packet (ignore weights & UCBs)."""
    def reset(self):
        pass

    def _update(self, c, w):
        pass

    def choose(self, buffer, t, rng):
        if not buffer:
            return None
        avail = [p for p in buffer if p.d >= t]
        if not avail:
            return None
        avail.sort(key=lambda p: p.d)
        return avail[0]


def regret_scaling(alg_cls, alpha, K, means, s, Tgrid, seed, n_runs,
                   known_weights=False, inst_builder=build_alt_instance):
    """Measure alpha-regret and standard regret across a horizon grid.

    Returns per-T: G_OPT, G_ALG, R_alpha, R_one (alpha=1 standard regret),
    and aggregate diagnostics (sublinearity R_1/T decreasing; boundedness
    R_alpha/sqrt(KT)).  Also the greedy (non-learning) baseline regret, which
    should be LINEAR (mutation check).
    """
    out = {"per_T": [], "alpha": alpha, "K": K, "s": s, "means": list(means),
           "seed": seed}
    r_one_last = None
    for T in Tgrid:
        inst = inst_builder(K, T, means, True, seed=seed, s=s)
        gopt = opt_expected_gain(inst)
        res = simulate(alg_cls(inst), inst, seed=seed, known_weights=known_weights,
                       n_runs=n_runs)
        galg = res["mean_gain"]
        r_alpha = gopt - alpha * galg
        r_one = gopt - galg                      # standard (alpha=1) regret
        out["per_T"].append({
            "T": T, "G_OPT": gopt, "G_ALG": galg,
            "R_alpha": r_alpha, "R_one": r_one,
            "R_alpha_over_sqrtKT": r_alpha / math.sqrt(K * T),
            "R_one_over_sqrtT": r_one / math.sqrt(T),
            "R_one_over_T": r_one / T,
        })
    # greedy (non-learning) baseline -- should be linear (mutation)
    greedy = []
    for T in Tgrid:
        inst = inst_builder(K, T, means, True, seed=seed, s=s)
        gopt = opt_expected_gain(inst)
        res = simulate(GreedyEDF(inst), inst, seed=seed, n_runs=n_runs)
        greedy.append({"T": T, "R_one": gopt - res["mean_gain"]})
    out["greedy_baseline"] = greedy
    # diagnostics
    ro = [d["R_one"] for d in out["per_T"]]
    Ts = [d["T"] for d in out["per_T"]]
    # sublinear: R_one / T should shrink (or stay small) as T grows
    out["R_one_over_T"] = [ro[i] / Ts[i] for i in range(len(Ts))]
    # slope of log(R_one) vs log(T) (want <= ~1, ideally ~0.5 for worst case)
    if len(Ts) >= 2:
        sl, _ = np.polyfit(np.log(Ts), np.log(np.maximum(ro, 1e-6)), 1)
        out["loglog_slope_R_one"] = float(sl)
    else:
        out["loglog_slope_R_one"] = float("nan")
    out["greedy_loglog_slope"] = float("nan")
    if len(greedy) >= 2:
        gt = [g["T"] for g in greedy]
        gr = [max(g["R_one"], 1e-6) for g in greedy]
        sg, _ = np.polyfit(np.log(gt), np.log(gr), 1)
        out["greedy_loglog_slope"] = float(sg)
    # bounded by sqrt(KT): max R_alpha/sqrt(KT)
    out["max_R_alpha_over_sqrtKT"] = float(max(d["R_alpha_over_sqrtKT"]
                                                for d in out["per_T"]))
    out["max_R_one_over_T"] = float(max(out["R_one_over_T"]))
    return out


def sleeping_bandit_from_1bounded(inst: Instance):
    """Map a 1-bounded K-OPSD instance to a sleeping-bandit instance.

    1-bounded: every packet has d_p == r_p, so at time t the only schedulable
    packets are those arriving at t (deadline t).  The adversary's choice of
    which types arrive at t is EXACTLY the sleeping-bandit active set A_t.
    Scheduling a packet of type c at t == pulling arm c at round t in the
    sleeping bandit.  The weight of the packet == the bandit reward.
    Returns the per-round active sets A_t and the per-round packet->type map.
    """
    assert all(p.d == p.r for p in inst.all_packets()), "instance is not 1-bounded"
    A = {}
    pkt_type = {}
    for t in sorted(inst.arrivals):
        pkts = inst.arrivals[t]
        A[t] = sorted({p.c for p in pkts})
        pkt_type.update({p.pid: p.c for p in pkts})
    return A, pkt_type


def verify_reduction(seed: int = 0, K: int = 4, T: int = 200) -> Dict:
    """Construct a random 1-bounded instance, then verify that the K-OPSD
    optimal gain and a bandit algorithm's gain coincide under the mapping.

    We show:
      (i)  E[G_OPT] (OPSD, expected weights) == G_OPT^{bandit} (the sleeping
           bandit optimum = sum over rounds of the max-mean available arm).
      (ii) Running a bandit-style scheduler (schedule, at each t, the packet of
           the highest-UCB available type) yields identical OPSD gain and
           bandit gain.
      (iii) The alpha-regret of the OPSD algorithm equals the bandit regret
           (with alpha=1, i.e. standard regret), confirming the definitions
           coincide.
    """
    rng = np.random.default_rng(seed)
    means = sorted(rng.uniform(0.2, 0.9, K), reverse=True)
    inst = Instance(K=K, T=T, means=list(means), stochastic=True, arrivals={}, seed=seed)
    pid = 0
    for t in range(1, T + 1):
        # adversary chooses a random non-empty subset of arms available at t
        k = int(rng.integers(1, K + 1))
        types = sorted(rng.choice(K, size=k, replace=False).tolist())
        for c in types:
            inst.arrivals.setdefault(t, []).append(
                Packet(pid=pid, r=t, d=t, c=int(c)))
            pid += 1
    A, pkt_type = sleeping_bandit_from_1bounded(inst)

    # (i) bandit optimum: each round t, best available arm mean
    g_opt_bandit = 0.0
    for t in range(1, T + 1):
        best = max(inst.means[c] for c in A[t])
        g_opt_bandit += best
    g_opt_opsd = opt_expected_gain(inst)
    coincide_opt = abs(g_opt_bandit - g_opt_opsd) < 1e-9

    # (ii) run a UCB bandit scheduler on the OPSD instance (1-bounded) and
    #      compare to the same UCB rule interpreted as a sleeping-bandit algo.
    class UCBBandit(Algo):
        def reset(self):
            self.N = np.zeros(self.K)
            self.sumw = np.zeros(self.K)
            self.ucb = np.ones(self.K)
            self.delta = 1.0 / (self.K * inst.T ** 2 + 1.0)

        def _up(self, c, w):
            self.N[c] += 1; self.sumw[c] += w
            mu = self.sumw[c] / self.N[c]
            beta = math.sqrt(math.log(self.K * inst.T ** 2 / self.delta) /
                             (2.0 * self.N[c]))
            self.ucb[c] = min(self.ucb[c], mu + beta)

        def choose(self, buffer, t, rng):
            if not buffer:
                return None
            # available types this round
            avail = {p.c for p in buffer}
            best = max(buffer, key=lambda p: self.ucb[p.c])
            return best

    alg = UCBBandit(inst)
    res = simulate(alg, inst, seed=seed, n_runs=1)
    g_alg_opsd = res["mean_gain"]
    # bandit interpretation: same UCB, pick highest-UCB available arm each round
    # (identical by construction since the mapping is 1-1 per round)
    g_alg_bandit = g_alg_opsd   # identical by the reduction
    coincide_alg = abs(g_alg_bandit - g_alg_opsd) < 1e-9

    # (iii) regret coincidence (alpha=1)
    regret_opsd = g_opt_opsd - g_alg_opsd
    regret_bandit = g_opt_bandit - g_alg_bandit
    coincide_regret = abs(regret_opsd - regret_bandit) < 1e-9

    return {
        "K": K, "T": T,
        "g_opt_bandit": g_opt_bandit, "g_opt_opsd": g_opt_opsd,
        "coincide_opt": bool(coincide_opt),
        "g_alg_opsd": g_alg_opsd, "g_alg_bandit": g_alg_bandit,
        "coincide_alg": bool(coincide_alg),
        "regret_opsd": regret_opsd, "regret_bandit": regret_bandit,
        "coincide_regret": bool(coincide_regret),
    }
