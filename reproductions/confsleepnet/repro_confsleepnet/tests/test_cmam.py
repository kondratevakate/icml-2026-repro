"""Formal check that CMAM fusion (Def. 2, Eq. 7-9) is a real evidential combination,
not numerically equivalent to average fusion and not degenerate at the conflict extremes.

Motivation: the CPU pilot showed CMAM not beating average on n=1 subject. Before
spending GPU budget we rule out a *code* cause — this test proves cmam_combine has the
qualitative behaviour an evidential fusion must have. Opinions are built from evidence
(u = K / S, always > 0), exactly as the model produces them, never from hand-set beliefs
(which can force the unreachable u = 0 boundary; see test_zero_uncertainty_boundary).

Run:  python -m pytest tests/test_cmam.py -q     (from repro_confsleepnet/)
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from model import K, cmam_combine, conflict_degree  # noqa: E402


def opinion(evidence):
    """Build (belief, u) from an evidence vector the way the model does (Sec. 3.4).

    alpha = e + 1, S = sum(alpha), belief = (alpha - 1) / S, u = K / S.
    Guarantees sum_k b + u = 1 and u > 0 for any finite evidence.
    """
    e = torch.tensor([evidence], dtype=torch.float32)
    alpha = e + 1
    S = alpha.sum(-1, keepdim=True)
    return (alpha - 1) / S, K / S


def test_normalization_holds():
    """sum_k b + u = 1 must survive fusion (Appendix A.5) for agree and conflict."""
    for ea, eb in ([[40, 1, 1, 0.5, 0.5], [35, 1.5, 1.5, 1, 0.5]],   # agree
                   [[40, 1, 1, 1, 0.5], [0.5, 1, 1, 1, 40]]):         # conflict
        ba, ua = opinion(ea)
        bb, ub = opinion(eb)
        b, u = cmam_combine(ba, ua, bb, ub)
        assert torch.isfinite(b).all() and torch.isfinite(u).all()
        assert (b >= 0).all() and (0 < u <= 1)
        assert abs(float(b.sum() + u) - 1.0) < 1e-5


def test_agreement_reduces_uncertainty():
    """Two confident opinions agreeing on a class must fuse to LOWER uncertainty than
    either input (evidence accumulation) — the property plain averaging cannot have."""
    ba, ua = opinion([40, 1, 1, 0.5, 0.5])
    bb, ub = opinion([35, 1.5, 1.5, 1, 0.5])
    b, u = cmam_combine(ba, ua, bb, ub)
    avg_u = (ua + ub) / 2
    assert float(u) < float(min(ua, ub)), "CMAM must sharpen on agreement"
    assert abs(float(u) - float(avg_u)) > 1e-3, "CMAM must differ from average fusion"
    assert int(b.argmax()) == 0, "agreed-upon class must remain the prediction"


def test_conflict_is_valid_and_higher_uncertainty():
    """Opposing near-one-hot opinions (C -> 1): output stays valid and carries MORE
    uncertainty than the agreement case — no NaN, no collapse to garbage."""
    ba, ua = opinion([40, 1, 1, 1, 0.5])
    bb, ub = opinion([0.5, 1, 1, 1, 40])
    assert float(conflict_degree(ba, bb)) > 0.9
    b, u = cmam_combine(ba, ua, bb, ub)
    assert torch.isfinite(b).all() and (0 < u <= 1)

    ba2, ua2 = opinion([40, 1, 1, 0.5, 0.5])
    bb2, ub2 = opinion([35, 1.5, 1.5, 1, 0.5])
    _, u_agree = cmam_combine(ba2, ua2, bb2, ub2)
    assert float(u) > float(u_agree), "conflict must be less certain than agreement"


def test_not_equivalent_to_average():
    """Across random opinion pairs, CMAM's fused belief must not equal the arithmetic
    mean of the two beliefs — otherwise ConfSleepNet and ConfSleepNet- are the same model."""
    torch.manual_seed(0)
    max_gap = 0.0
    for _ in range(50):
        ba, ua = opinion((torch.rand(K) * 30).tolist())
        bb, ub = opinion((torch.rand(K) * 30).tolist())
        b, _ = cmam_combine(ba, ua, bb, ub)
        max_gap = max(max_gap, float((b - (ba + bb) / 2).abs().max()))
    assert max_gap > 1e-2, "CMAM collapses to average fusion — the ablation would be void"


def test_zero_uncertainty_boundary():
    """Documented boundary: at u = 0 exactly (infinite evidence) the ua+ub denominator
    vanishes and fusion degenerates to b = 0, u = 1. This is UNREACHABLE in training
    (u = K / S > 0 for finite evidence) but is pinned here so the guard is intentional,
    not accidental. If model output ever hits u = 0, this fusion must be revisited."""
    b_onehot = torch.tensor([[1.0, 0, 0, 0, 0]])
    u_zero = torch.zeros(1, 1)
    b, u = cmam_combine(b_onehot, u_zero, b_onehot, u_zero)
    assert float(u) == 1.0 and float(b.sum()) == 0.0
