"""LERD encoder + classifier (Appendix F.2, F.5).

This is the "No prior" LERD variant: the EEGNet-style temporal-spatial encoder
feeding the classifier directly, without the dLIF / ERG priors. Table 4 reports
it at 70.52 +/- 11.83 % accuracy on Cohort A, so reproducing this number is the
first checkpoint before the priors are layered on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

C = 19  # electrodes
T = 1000  # samples per 2 s window


class Encoder(nn.Module):
    """Temporal-spatial factorized encoder (EEGNet-style, F.2)."""

    def __init__(self, F1=8, D=2, drop=0.1):
        super().__init__()
        F2 = F1 * D
        # Block 1: temporal depthwise -> spatial depthwise
        self.temporal = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F2, (C, 1), groups=F1, bias=False)  # depthwise over electrodes
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(drop)
        # Block 2: depthwise-separable temporal
        self.sep_dw = nn.Conv2d(F2, F2, (1, 16), groups=F2, padding=(0, 8), bias=False)
        self.sep_pw = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(drop)
        self.act = nn.ELU()
        self.out_dim = F2 * (T // 4 // 8)

    def forward(self, x):                 # x: (B, 1, 19, 1000)
        x = self.bn1(self.temporal(x))
        x = self.act(self.bn2(self.spatial(x)))   # (B, F2, 1, T)
        x = self.drop1(self.pool1(x))
        x = self.sep_pw(self.sep_dw(x))
        x = self.act(self.bn3(x))
        x = self.drop2(self.pool2(x))     # (B, F2, 1, T//32)
        return x.flatten(1)               # (B, out_dim)


class LERDNoPrior(nn.Module):
    def __init__(self, n_classes=3, hidden=128, drop=0.1):
        super().__init__()
        self.enc = Encoder(drop=drop)
        self.head = nn.Sequential(
            nn.Linear(self.enc.out_dim, hidden), nn.ELU(), nn.Dropout(drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.head(self.enc(x))
