"""Infrastructure smoke test (not a claim check): does the pipeline run on CPU, and how long do steps take?"""
import sys
import time

import torch

sys.path.insert(0, "/work/checks")
from _common import *  # noqa: E402,F401,F403
from _common import U, DATA, DEVICE, SynthDL, load_config, load_masks, load_network, prepare_case, make_attack, log

kx, ky = int(sys.argv[1]), int(sys.argv[2])
cfg = load_config(kx, ky)
om, ds = load_masks()
net = load_network()
dl = torch.utils.data.DataLoader(SynthDL(DATA + "/"), batch_size=1, shuffle=False, num_workers=0)
ksp, coil, name = next(iter(dl))
ksp, coil = ksp.to(DEVICE), coil.to(DEVICE)
log("shapes ksp", tuple(ksp.shape), ksp.dtype, "coil", tuple(coil.shape), "mask", tuple(om.shape), om.dtype)
seed_everything = U.seed_everything
seed_everything(0)
t = time.time()
zf, label, jitt = prepare_case(ksp, coil, om, net, cfg)
log("prepare_case (1 fwd for jitter):", round(time.time() - t, 2), "s")
t = time.time()
with torch.no_grad():
    rec = net(zf, coil, om)
log("1 forward:", round(time.time() - t, 2), "s")
t = time.time()
zf.requires_grad_(True)
rec = net(zf, coil, om)
l = U.L2_Loss(rec, label)
l.backward()
log("1 forward+backward:", round(time.time() - t, 2), "s")
import numpy as np
psnr_zf = U.getpsnr(np.abs(zf.detach().squeeze().numpy()), np.abs(label.detach().squeeze().numpy()))
psnr_rec = U.getpsnr(np.abs(rec.detach().squeeze().numpy()), np.abs(label.detach().squeeze().numpy()))
log("clean PSNR zero-filled", round(float(psnr_zf), 2), "clean PSNR recon", round(float(psnr_rec), 2))
t = time.time()
cfg2 = load_config(kx, ky)
cfg2["Attack"]["iterations"] = 2
zf2 = zf.detach().clone().requires_grad_(True)
zp = make_attack(zf2, net, om, coil, cfg2, 0.01)
log("attack, 2 iterations:", round(time.time() - t, 2), "s")
t = time.time()
cfg3 = load_config(kx, ky, iterations_mitigate=1)
PSNR, SSIM, LOSS = U.Cyclic_Mitigation(cfg3, net, zp, label.clone(), coil, om, ds, DEVICE, jitt)
log("Cyclic_Mitigation, 1 iteration:", round(time.time() - t, 2), "s; psnr list len", len(PSNR))
import resource
log("peak RSS MB:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "threads", torch.get_num_threads())
