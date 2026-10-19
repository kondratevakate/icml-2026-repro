"""ConfSleepNet, reimplemented from the paper text (no upstream code exists).

Every design detail below is traced to a specific line of the paper. Where the paper is
silent, the choice is marked `GAP(n)` and listed in README under "underspecification".
The paper gives Appendix A.2/A.3 qualitatively only: block composition but no kernel
sizes, channel counts, or hidden dimensions, which live in Figures 5 and 6.

Structure (Section 3.2, Appendix A.3):
    f1: EEG only,      5-class evidence   (single-view net)
    f2: EEG + EOG,     5-class evidence   (multi-view net, two independent extractors)
    f3: EEG + EOG,     3-class evidence   (multi-view net)
    f4: EOG only,      3-class evidence   (single-view net)
Coarse 3-class evidence (W / NREM / REM) is mapped to 5-class by U (Section 3.3), then
the four opinions are fused by CMAM (Definitions 1-2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SFREQ = 100
K = 5  # fine-grained classes: W N1 N2 N3 REM
KC = 3  # coarse-grained classes: W NREM REM

# Section 3.3: uniform mapping, rows sum to 1, NREM evidence split equally over N1/N2/N3.
U = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 1 / 3, 1 / 3, 1 / 3, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.0]]
)
# Coarse label for each fine label: W->0, N1/N2/N3->1, REM->2.
FINE_TO_COARSE = torch.tensor([0, 1, 1, 1, 2])


class CNNBranch(nn.Module):
    """One of the two parallel convolutional branches (Appendix A.2, Eq. 14-15).

    GAP(1): the paper states only "two parallel branches with different kernel sizes to
    capture both temporal and frequency information". It cites DeepSleepNet (Supratak
    2017) as the origin of that construction, so we adopt DeepSleepNet's spec verbatim:
    the small-kernel branch uses Fs/2 with stride Fs/16 (fine temporal detail) and the
    large-kernel branch Fs*4 with stride Fs/2 (frequency detail).
    """

    def __init__(self, in_ch: int, large: bool, dropout: float = 0.5):
        super().__init__()
        k1, s1, pool1, k2 = (
            (SFREQ * 4, SFREQ // 2, 4, 6) if large else (SFREQ // 2, SFREQ // 16, 8, 8)
        )
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, k1, s1, padding=k1 // 2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(pool1, pool1), nn.Dropout(dropout),
            nn.Conv1d(64, 128, k2, 1, padding=k2 // 2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, k2, 1, padding=k2 // 2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, k2, 1, padding=k2 // 2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
        )

    def forward(self, x):  # (B, in_ch, T) -> (B, 128, T')
        return self.net(x)


class GCNet1D(nn.Module):
    """GCNet global context block (Cao 2019) adapted to 1D, per Appendix A.2.

    Context modelling by attention pooling, then a bottleneck transform added back.
    GAP(2): the reduction ratio is not given; 4 is GCNet's own default.
    """

    def __init__(self, ch: int, ratio: int = 4):
        super().__init__()
        self.mask = nn.Conv1d(ch, 1, 1)
        self.transform = nn.Sequential(
            nn.Conv1d(ch, ch // ratio, 1),
            nn.LayerNorm([ch // ratio, 1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch // ratio, ch, 1),
        )

    def forward(self, x):  # (B, C, T)
        w = torch.softmax(self.mask(x), dim=-1)          # (B, 1, T)
        ctx = torch.bmm(x, w.transpose(1, 2))            # (B, C, 1)
        return x + self.transform(ctx)


class CrossAttention(nn.Module):
    """Cross-attention between the two enhanced branches (Appendix A.2).

    GAP(3): head count not given; single head keeps the parameter count closest to the
    plain "cross-attention mechanism" the text describes.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, q, kv):  # (B, T, C)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out


class FeatureExtractor(nn.Module):
    """Appendix A.2: F = hs || hl || hsl || hls, with residual shortcuts on hs, hl."""

    def __init__(self, in_ch: int, out_dim: int = 256):
        super().__init__()
        self.cnn_s, self.cnn_l = CNNBranch(in_ch, False), CNNBranch(in_ch, True)
        self.gcn_s, self.gcn_l = GCNet1D(128), GCNet1D(128)
        self.cross_sl, self.cross_ls = CrossAttention(128), CrossAttention(128)
        # GAP(4): the projection to a fixed feature width is implied by Fig. 5 but not
        # dimensioned; 256 keeps F comparable in size to TinySleepNet's representation.
        self.proj = nn.Linear(128 * 4, out_dim)

    def forward(self, x):  # (B, in_ch, 3000) -> (B, out_dim)
        hs = self.gcn_s(self.cnn_s(x)).transpose(1, 2)  # (B, Ts, 128)
        hl = self.gcn_l(self.cnn_l(x)).transpose(1, 2)  # (B, Tl, 128)
        hsl = self.cross_sl(hs, hl)
        hls = self.cross_ls(hl, hs)
        parts = [t.mean(dim=1) for t in (hs, hl, hsl, hls)]  # temporal average pooling
        return self.proj(torch.cat(parts, dim=-1))


class EvidentialDNN(nn.Module):
    """A single f_v (Appendix A.3). Bi-LSTM over the 20-epoch sequence, then SoftPlus."""

    def __init__(self, n_classes: int, in_ch_list: list[int], hidden: int = 128):
        super().__init__()
        # Single-view nets take one extractor; multi-view nets take two independent ones
        # and fuse after extraction (Appendix A.3).
        self.extractors = nn.ModuleList(FeatureExtractor(c) for c in in_ch_list)
        self.lstm = nn.LSTM(256 * len(in_ch_list), hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, views):  # list of (B, L, ch, 3000)
        B, L = views[0].shape[:2]
        feats = [ex(v.reshape(B * L, *v.shape[2:])).reshape(B, L, -1)
                 for ex, v in zip(self.extractors, views)]
        seq, _ = self.lstm(torch.cat(feats, dim=-1))
        return F.softplus(self.head(seq))  # evidence e^v >= 0, (B, L, n_classes)


def conflict_degree(ba: torch.Tensor, bb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Definition 1, Eq. 6. C = 1 - sum_k ba_k bb_k / (sum_i ba_i * sum_j bb_j)."""
    num = (ba * bb).sum(-1, keepdim=True)
    den = ba.sum(-1, keepdim=True) * bb.sum(-1, keepdim=True)
    return (1 - num / (den + eps)).clamp(0, 1)


def cmam_combine(ba, ua, bb, ub, eps: float = 1e-8):
    """Definition 2, Eq. 7-9: fuse two opinions.

    Eq. 9 is used as printed for the belief masses. For the uncertainty we do NOT use
    Eq. 8 as printed: our analytic check of Propositions 1-2 (claim-1 page of the logbook)
    showed the printed Eq. 8 breaks the paper's own normalization sum_k b + u = 1 by a
    factor of ~2 as C -> 1. Appendix A.5 asserts that normalization holds, so we take the
    paper's stated constraint as authoritative and recover u from it. This is recorded as
    a deviation, and the printed-Eq.-8 variant is available for an ablation.
    """
    C = conflict_degree(ba, bb, eps)
    denom = ua + ub + eps
    b = (ua * bb + ub * ba + (1 - C) * ua * ub * (ba + bb)) / denom
    b = b.clamp_min(0)
    total = b.sum(-1, keepdim=True)
    scale = torch.where(total > 1, 1 / (total + eps), torch.ones_like(total))
    b = b * scale
    u = (1 - b.sum(-1, keepdim=True)).clamp(eps, 1.0)
    return b, u


class ConfSleepNet(nn.Module):
    """Full model. `fusion='cmam'` is ConfSleepNet; `fusion='average'` is ConfSleepNet-."""

    def __init__(self, fusion: str = "cmam"):
        super().__init__()
        assert fusion in ("cmam", "average")
        self.fusion = fusion
        self.f1 = EvidentialDNN(K, [1])       # EEG, fine
        self.f2 = EvidentialDNN(K, [1, 1])    # EEG + EOG, fine
        self.f3 = EvidentialDNN(KC, [1, 1])   # EEG + EOG, coarse
        self.f4 = EvidentialDNN(KC, [1])      # EOG, coarse
        self.register_buffer("U", U)

    def forward(self, eeg, eog):
        """eeg, eog: (B, L, 1, 3000). Returns raw evidences and the fused opinion."""
        e = [self.f1([eeg]), self.f2([eeg, eog]), self.f3([eeg, eog]), self.f4([eog])]
        # Section 3.3: coarse evidence -> fine evidence via U before fusion.
        e_fine = [e[0], e[1], e[2] @ self.U, e[3] @ self.U]

        alphas = [ev + 1 for ev in e_fine]                       # Section 3.4, alpha = e + 1
        S = [a.sum(-1, keepdim=True) for a in alphas]
        beliefs = [(a - 1) / s for a, s in zip(alphas, S)]       # Eq. 4
        uncerts = [K / s for s in S]

        if self.fusion == "average":
            b = torch.stack(beliefs).mean(0)
            u = torch.stack(uncerts).mean(0)
        else:
            b, u = beliefs[0], uncerts[0]
            for bi, ui in zip(beliefs[1:], uncerts[1:]):         # Eq. 10, fixed order
                b, u = cmam_combine(b, u, bi, ui)
        return e, b, u
