"""C00751 - side observation only (outcome stays not_checkable_on_synthetic, see checks/plan.json).

Instantiates the repo's ViT encoders and counts parameters, to see whether the '1M ... 85M' ladder of the
scaling analysis exists in the released code. This says nothing about whether performance scales.
"""
import sys
sys.path.insert(0, "/work/work/repo")
import torch
from osf.backbone import vit1d_cls as V

for name in ["vit_nano", "vit_tiny", "vit_small", "vit_middle", "vit_base"]:
    for pt, lw in [(128, 1), (64, 0), (64, 1)]:
        m = getattr(V, name)(num_leads=12, seq_len=1920, patch_size=pt, lead_wise=lw, patch_size_ch=4)
        n = sum(p.numel() for p in m.parameters())
        n_blocks = sum(p.numel() for k, p in m.named_parameters() if k.startswith("block"))
        print(f"{name:11s} patch_t={pt} lead_wise={lw}: total {n/1e6:7.2f} M | transformer blocks only {n_blocks/1e6:7.2f} M")
