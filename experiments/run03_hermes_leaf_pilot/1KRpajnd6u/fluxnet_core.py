"""fluxnet_core.py — faithful numpy re-implementation of the FluxNet update and heads.

Sources (notes_paper.md):
  Eq. 2/6  conservative transport update
  Sec 3.3  L-head / U-head parameterizations
  Eq. 3    D-head averaged update
All periodic; inflow obtained by rolling the outgoing flux field (as the paper states).
No network is needed: Prop 1-3 hold for ANY flux field, so we sample the head *outputs*
(logits) randomly, which is a strictly stronger test than testing one trained net.
"""
import numpy as np


def stencil(radius, ndim):
    """Symmetric stencil: all offsets in [-r,r]^ndim except 0. Symmetric by construction."""
    rng = range(-radius, radius + 1)
    if ndim == 1:
        offs = [(d,) for d in rng if d != 0]
    else:
        offs = [(a, b) for a in rng for b in rng if not (a == 0 and b == 0)]
    return offs


def inflow_from_outflow(F, offs):
    """F[k] = outflow field for direction offs[k] (amount leaving cell i towards i+d).
    Inflow to cell i along direction d comes from cell i-d, i.e. roll by +d."""
    tot = np.zeros_like(F[0])
    for k, d in enumerate(offs):
        tot = tot + np.roll(F[k], shift=d, axis=tuple(range(len(d))))
    return tot


def update(u, F, offs):
    """Eq. 2: u - total_outflow + total_inflow."""
    return u - np.sum(F, axis=0) + inflow_from_outflow(F, offs)


def _softmax(x, axis=0):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def l_head_flux(u, l, a_logit, pi_logits, offs, alpha_cap=1.0):
    """F_{i->i+d} = (u_i - l) * alpha_i * pi_{i->i+d};  alpha=sigmoid in (0,1)."""
    a = u - l
    alpha = alpha_cap * _sigmoid(a_logit)
    pi = _softmax(pi_logits, axis=0)
    return (a * alpha)[None, ...] * pi


def u_head_influx(u, umax, b_logit, rho_logits, offs, beta_cap=1.0):
    """Inflow-parameterized: G[k] = amount cell i pulls from neighbour i-... along dir k.
    F_{j->i} = (umax - u_i) * beta_i * rho_{j->i};  sum_k rho = 1.
    Returns G with G[k][i] = inflow to i arriving along direction offs[k]."""
    b = umax - u
    beta = beta_cap * _sigmoid(b_logit)
    rho = _softmax(rho_logits, axis=0)
    return (b * beta)[None, ...] * rho


def u_head_update(u, G, offs):
    """Inflow field G -> equivalent outflow field for the *sender*.
    G[k][i] is pulled from cell i - offs[k]; so sender s = i - d contributes G[k][s+d].
    Outflow field seen by sender: O[k] = roll(G[k], -d)."""
    inflow = np.sum(G, axis=0)
    outflow = np.zeros_like(u)
    for k, d in enumerate(offs):
        outflow = outflow + np.roll(G[k], shift=tuple(-x for x in d), axis=tuple(range(len(d))))
    return u + inflow - outflow


def d_head_update(u, l, umax, a_logit, pi_logits, b_logit, rho_logits, offs):
    """Eq. 3: average of the two branch state-changes. Returns (u_next, du_out, du_in)."""
    Fout = l_head_flux(u, l, a_logit, pi_logits, offs)
    du_out = update(u, Fout, offs) - u
    G = u_head_influx(u, umax, b_logit, rho_logits, offs)
    du_in = u_head_update(u, G, offs) - u
    return u + 0.5 * (du_out + du_in), du_out, du_in
