"""Bring-up only: where does a pre-training step spend its time?"""
import time, torch
import rovtl_common as rc
rc.init(reduced=True); rc.patch_resnet_tokens_for_small_inputs()
from rovtl_common import *
D = "/work/work/data/tiny16"
prep = make_pretrain_prepper(D)
ds = MRContrastiveDatasetH5(D + "/train.h5", augmentation_rate=-1, simulate_missing=True)
data = DataCollator(prep)([ds[i] for i in range(32)])
model = ImageTabularPretraining(dim=3); model.train()
def tm(name, fn, n=2):
    for k in range(n):
        t = time.time(); fn(); log(name, round(time.time() - t, 2), "s")
tm("resnet fwd", lambda: model.image_encoder(data["image"].float()))
tm("resnet fwd+bwd", lambda: sum(o.sum() for o in model.image_encoder(data["image"].float())[0]).backward())
tm("tarte fwd", lambda: model.tabular_encoder(data["x"], data["edge_attr"], data["mask"]))
tm("tarte fwd+bwd", lambda: sum(o.sum() for o in model.tabular_encoder(data["x"], data["edge_attr"], data["mask"])).backward())
import torch
log("threads", torch.get_num_threads())
for frames in (10, 4):
    x = torch.randn(32, 11, frames, 16, 16)
    tm(f"resnet fwd+bwd frames={frames}", lambda: sum(o.sum() for o in model.image_encoder(x)[0]).backward(), n=1)
