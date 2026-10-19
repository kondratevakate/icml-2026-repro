"""Bring-up only: final feature-map shape of the repo 3D ResNet-50 for several scan shapes, and monai conv1 strides."""
import torch
import rovtl_common as rc
rc.init(reduced=True)
from rovtl_common import ResNetTokens
m = ResNetTokens(dim=3).eval()
print("conv1", m.resnet.conv1 if hasattr(m.resnet, "conv1") else [n for n, _ in m.resnet.named_children()][:4], flush=True)
for shape in [(10, 16, 16), (10, 33, 33), (20, 16, 16), (34, 8, 8), (10, 64, 64), (24, 8, 8), (40, 8, 8)]:
    x = torch.randn(2, 11, *shape)
    with torch.no_grad():
        out = m.resnet(x)[-1]
    print(shape, "final map", tuple(out.shape), flush=True)
