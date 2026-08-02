"""Shared implementations for the reproduction of DUmWdZetqZ.

Everything here is a verbatim transcription of the pseudocode in notes_paper.md:
  - Algorithm 3 (FC2FB), Algorithm 4 (FCW2S), Algorithm 5 (PE-KHN), Algorithm 6 (FC2AT)
  - the bounds of Theorem 3.2, Prop 4.2/4.3, Theorem D.5, Corollary 5.2.

Only numpy + stdlib.
"""
import math
import numpy as np

# ----------------------------------------------------------------------------- bounds
def thm32_bound(B, A, Q, delta0):
    """Theorem 3.2:  3 exp( -B / (4Q/ln(1/delta0) + 4 log2(B/Q) A) )."""
    denom = 4.0 * Q / math.log(1.0 / delta0) + 4.0 * math.log2(B / Q) * A
    return 3.0 * math.exp(-B / denom)


def thm32_budget_condition(A, C, Q, delta0):
    """Minimum budget B for which Theorem 3.2 applies."""
    inner = 2.0 * A / Q * math.log(1.0 / delta0) + 2.0 * (C + 1.0) / Q
    return 2.0 * (A * math.log(1.0 / delta0) + (C + 1.0)) * math.log(inner)


def thmD5_bound(T, A, Q, delta0):
    """Theorem D.5:  3 exp( -T / (16Q/ln(1/delta0) + 16 log2(T/Q) A) )."""
    denom = 16.0 * Q / math.log(1.0 / delta0) + 16.0 * math.log2(T / Q) * A
    return 3.0 * math.exp(-T / denom)


def cor52_bound(B, A):
    """Corollary 5.2 as printed:  3 exp( -B / (4 + 4 A ln B) )."""
    return 3.0 * math.exp(-B / (4.0 + 4.0 * A * math.log(B)))


def prop42_bound(L, delta0):
    """Prop 4.2:  (4 e delta0)^(L/4)."""
    return math.exp(-(L / 4.0) * math.log(1.0 / (4.0 * math.e * delta0)))


def prop43_bound(L, delta0):
    """Prop 4.3:  (2 e delta0)^(L/2)."""
    return math.exp(-(L / 2.0) * math.log(1.0 / (2.0 * math.e * delta0)))


# ------------------------------------------------------- worst-case strong FC oracle
class StrongFCOracle:
    """An FC algorithm that saturates Definition 3.1 with constants (A, C).

    On input delta (<= 1/2) it behaves as:
      * w.p. delta      : self-terminates at time 1 outputting a WRONG arm
                          -> P(Jhat != 1, tau < inf) = delta  (delta-correct, tight)
      * w.p. delta      : never self-terminates
                          -> P(tau > T*_delta) = delta        (tight)
      * w.p. 1 - 2delta : self-terminates at time ceil(T*_delta) with the correct arm
    with T*_delta = A ln(1/delta) + C. This is the adversarial member of Definition 3.1:
    any theorem that holds "for a strong FC algorithm" must hold for it.
    """

    def __init__(self, A, C):
        self.A = float(A)
        self.C = float(C)

    def tstar(self, log_inv_delta):
        return self.A * log_inv_delta + self.C

    def outcome_probs(self, log_inv_delta, budget_cap):
        """Exact distribution of one FC2FB stage, given cap B'.

        Returns (p_stop_wrong, p_stop_correct, p_no_output) using log(1/delta).
        """
        delta = math.exp(-log_inv_delta)
        if delta > 0.5:
            raise ValueError("oracle requires delta <= 1/2")
        need = math.ceil(self.tstar(log_inv_delta))
        p_wrong = delta                       # stops at time 1, always fits in the cap
        if need <= budget_cap:
            return p_wrong, 1.0 - 2.0 * delta, delta
        return p_wrong, 0.0, 1.0 - delta      # correct path force-terminated

    def run(self, log_inv_delta, budget_cap, rng):
        """Monte-Carlo single run. Returns (terminated, correct)."""
        delta = math.exp(-log_inv_delta)
        u = rng.random()
        if u < delta:
            return True, False                                     # early wrong stop
        if u < 2.0 * delta:
            return False, False                                    # never terminates
        return (math.ceil(self.tstar(log_inv_delta)) <= budget_cap), True


# ------------------------------------------------------------------ Algorithm 3
def fc2fb_schedule(B, Q):
    """R, B' of Algorithm 3."""
    R = int(math.floor(math.log2(B / Q)))
    if R < 1:
        R = 1
    Bp = int(math.floor(B / R))
    return R, Bp


def fc2fb_error_exact(oracle, B, Q, delta0, schedule="paper"):
    """Exact P(Jhat != 1) of Algorithm 3 driven by StrongFCOracle. No sampling.

    schedule="paper"    : L_r = 2^(R-r)                       (Algorithm 3)
    schedule="constant" : L_r = 1 for all r                   (MUTATION)
    schedule="reversed" : L_r = 2^(r-1)                       (MUTATION)
    The arbitrary fallback arm is counted as wrong (worst case, K >= 2).
    """
    R, Bp = fc2fb_schedule(B, Q)
    log_inv_d0 = math.log(1.0 / delta0)
    reach, err = 1.0, 0.0
    for r in range(1, R + 1):
        if schedule == "paper":
            L = 2.0 ** (R - r)
        elif schedule == "constant":
            L = 1.0
        elif schedule == "reversed":
            L = 2.0 ** (r - 1)
        else:
            raise ValueError(schedule)
        pw, pc, pn = oracle.outcome_probs(L * log_inv_d0, Bp)
        err += reach * pw
        reach *= pn
    err += reach          # never produced an output -> arbitrary (wrong) arm
    return err


def fc2fb_error_mc(oracle, B, Q, delta0, n, seed, schedule="paper"):
    rng = np.random.default_rng(seed)
    R, Bp = fc2fb_schedule(B, Q)
    log_inv_d0 = math.log(1.0 / delta0)
    wrong = 0
    for _ in range(n):
        got = None
        for r in range(1, R + 1):
            if schedule == "paper":
                L = 2.0 ** (R - r)
            elif schedule == "constant":
                L = 1.0
            else:
                L = 2.0 ** (r - 1)
            term, corr = oracle.run(L * log_inv_d0, Bp, rng)
            if term:
                got = corr
                break
        if got is not True:
            wrong += 1
    return wrong / n


# ------------------------------------------------------------------ Algorithm 5
def pe_khn(mu, sigma, delta, rng, budget=None):
    """Algorithm 5 (phased elimination, known heterogeneous noise).

    Returns (terminated, correct, samples_used). Arm 0 is the best arm.
    If `budget` is given, the run is force-terminated when it is exhausted.
    """
    K = len(mu)
    S = list(range(K))
    total = np.zeros(K, dtype=np.int64)
    ssum = np.zeros(K)
    used = 0
    ell = 1
    while len(S) > 1:
        eps = 1.0 / (2.0 ** ell)
        dl = delta / (ell * (ell + 1.0))
        want = {i: math.ceil(2.0 * sigma[i] ** 2 / eps ** 2 * math.log(K / dl)) for i in S}
        extra = sum(max(0, want[i] - total[i]) for i in S)
        if budget is not None and used + extra > budget:
            return False, False, budget
        for i in S:
            n_new = want[i] - total[i]
            if n_new > 0:
                ssum[i] += rng.normal(mu[i], sigma[i], size=n_new).sum()
                total[i] += n_new
                used += n_new
        means = {i: ssum[i] / total[i] for i in S}
        best = max(means[i] for i in S)
        S = [i for i in S if means[i] > best - 2.0 * eps]
        ell += 1
        if ell > 60:
            break
    return True, (S[0] == 0), used


def pe_khn_constants(mu, sigma):
    """A and C of Definition 3.1 for PE-KHN, read off Theorem 5.1."""
    K = len(mu)
    gaps = [mu[0] - m for m in mu]
    D2 = gaps[1]
    coeffs = [64.0 * sigma[0] ** 2 / D2 ** 2] + \
             [64.0 * sigma[j] ** 2 / gaps[j] ** 2 for j in range(1, K)]
    Ds = [D2] + [gaps[j] for j in range(1, K)]
    A = sum(coeffs)
    C = sum(c * math.log(4.0 * K * (math.log(2.0) ** 2) * math.log(4.0 / d) ** 2)
            for c, d in zip(coeffs, Ds))
    return A, C


def tstar_thm51(mu, sigma, delta):
    A, C = pe_khn_constants(mu, sigma)
    return A * math.log(1.0 / delta) + C
