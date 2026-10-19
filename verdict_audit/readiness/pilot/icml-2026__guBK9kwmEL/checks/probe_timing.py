"""Bring-up only (not a claim check): time one pre-training step, one fine-tuning step and one evaluation pass on tiny16."""
import time, torch
import rovtl_common as rc
rc.init(reduced=True); rc.patch_resnet_tokens_for_small_inputs()
from rovtl_common import *
D = "/work/work/data/tiny16"
t = time.time(); prep = make_pretrain_prepper(D); log("pretrain prepper fit", round(time.time() - t, 1), "s")
ds = MRContrastiveDatasetH5(D + "/train.h5", augmentation_rate=-1, simulate_missing=True)
col = DataCollator(prep)
batch = [ds[i] for i in range(32)]
t = time.time(); data = col(batch); log("collate 32 rows", round(time.time() - t, 2), "s", tuple(data["x"].shape), tuple(data["image"].shape))
model = ImageTabularPretraining(dim=3); model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = CLIPLoss(0.1)
for k in range(2):
    t = time.time()
    o1, o2, _ = model(data["x"], data["edge_attr"], data["mask"], data["image"].float())
    loss = sum(crit(o, o2[i])[0] for i, o in enumerate(o1)); opt.zero_grad(); loss.backward(); opt.step()
    log("pretrain step", k, round(time.time() - t, 2), "s loss", float(loss))
dprep = make_downstream_prepper(D)
dds = MRDownstreamDatasetH5(h5_path=D + "/train.h5", filter_csv=D + "/train.csv", simulate_missing=True, sample_more=0.3, augmentation_rate=-1)
dcol = DownstreamCollator(dprep)
t = time.time(); dd = dcol([dds[i] for i in range(32)]); log("down collate 32 (64 rows)", round(time.time() - t, 2), tuple(dd["x"].shape))
torch.save(model.state_dict(), "/tmp/probe.pth")
clf = load_classifier("/tmp/probe.pth", True); clf.train()
o = torch.optim.AdamW(clf.parameters(), lr=1e-4)
for k in range(2):
    t = time.time()
    lg, il, tl = clf(dd["x"], dd["edge_attr"], dd["mask"], dd["image"].float())
    l = nn.BCEWithLogitsLoss()(lg, dd["label"]); o.zero_grad(); l.backward(); o.step()
    log("finetune step", k, round(time.time() - t, 2), "s", tuple(lg.shape))
t = time.time(); a = eval_auc(clf, D, dprep, 0.5, 1); log("eval_auc f=0.5 (128 test)", round(time.time() - t, 1), "s auc", a)
t = time.time(); img, tab = encode_test(model, D, prep, [1.0]); log("encode_test f=1.0 (128)", round(time.time() - t, 1), "s")
