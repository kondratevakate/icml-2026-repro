"""Shared helpers for the CycMit-MRI pilot checks. Runs inside the container (paths under /work).

Everything algorithmic comes from the repository (src/Utils.py, src/Unrolled_Network.py, DataLoader.py).
The only non-repository logic is the glue that Mitigation.py keeps inside its __main__ block
(zero-filled image, label, normalisation, noise jittering, attack) which is replicated line by line here.
"""
import glob
import hashlib
import os
import sys

import numpy as np
import scipy.io as sio
import torch
import yaml

REPO = "/work/work/repo"
DATA = "/work/work/data"
sys.path.insert(0, REPO)

import importlib.util  # noqa: E402
import types  # noqa: E402

# --- one-line CPU-compatibility patch (documented in results.json notes) -------------------------------
# On CPU, Cyclic_Mitigation fails with "A view was created in no_grad mode and its base ... modified
# inplace" because recon_i is a no_grad *view* of Recons_mit that is later modified in place and then
# viewed again via .cpu() (a no-op on CPU; it is a copy on the CUDA device the authors used).
# We load src/Utils.py with exactly one textual change: recon_i becomes an independent copy (same values).
_src = open(f"{REPO}/src/Utils.py").read()
_old = "recon_i    = Recons_mit[:,:,0]"
assert _src.count(_old) == 1, "patch anchor not found exactly once"
_src = _src.replace(_old, "recon_i    = Recons_mit[:,:,0].detach().clone()")
_pkg = types.ModuleType("src"); _pkg.__path__ = [f"{REPO}/src"]; sys.modules["src"] = _pkg
U = types.ModuleType("src.Utils"); U.__file__ = f"{REPO}/src/Utils.py"; sys.modules["src.Utils"] = U
exec(compile(_src, f"{REPO}/src/Utils.py", "exec"), U.__dict__)
from src.Utils import (CG_EEH, IFFT, fft_loss, noise_jiterring, seed_everything,  # noqa: E402
                       attack_generation)
from src.Unrolled_Network import UnrolledNet  # noqa: E402
from DataLoader import DataLoaderSL  # noqa: E402

# plotting inside Cyclic_Mitigation writes PNGs every iteration; it does not touch the algorithm
U.Imshow_Custom = lambda *a, **k: None

torch.set_num_threads(int(os.environ.get("P9_THREADS", "1")))
DEVICE = torch.device("cpu")


class SynthDL(DataLoaderSL):
    """Repo loader (same keys, same transposes); only the glob is widened from kneePD.mat to our files."""

    def __init__(self, path, pattern="case_*.mat"):
        self.datapath = sorted(glob.glob(path + pattern))


def load_config(kx, ky, nc=15, **mit_overrides):
    with open(os.path.join(REPO, "Config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["device"] = "cpu"
    cfg["kx"], cfg["ky"], cfg["nc"] = kx, ky, nc
    cfg["Saving"]["path"] = "/work/work/out"
    cfg["Mitigation"].update(mit_overrides)
    os.makedirs(cfg["Saving"]["path"], exist_ok=True)
    os.makedirs(cfg["Saving"]["path"] + "/iterations", exist_ok=True)
    return cfg


def load_masks():
    om = torch.tensor(sio.loadmat(f"{DATA}/omega_mask.mat")["mask"].transpose([1, 0])).unsqueeze(0).to(DEVICE)
    ds = [torch.tensor(sio.loadmat(f"{DATA}/delta{i}.mat")["mask"].transpose([1, 0]), dtype=torch.complex64)
          .unsqueeze(0).to(DEVICE) for i in (1, 2, 3)]
    return om, ds


def load_network():
    net = UnrolledNet(mu=0.1813, Unrolls=10)  # exactly as Mitigation.py
    net.load_state_dict(torch.load(f"{REPO}/BestModel/checkpoint.pth", map_location="cpu", weights_only=True))
    net.to(DEVICE)
    net.eval()
    return net


def param_hash(net):
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare_case(ksp, coil, om, net, cfg):
    """Replicates Mitigation.py (zero-filled / label / scale / noise jittering)."""
    zero_filled = IFFT(ksp * om)
    label = IFFT(ksp)
    zero_filled = torch.sum(zero_filled * torch.conj(coil), axis=1, keepdims=True)
    label = torch.sum(label * torch.conj(coil), axis=1, keepdims=True)
    scale = torch.max(torch.abs(zero_filled))
    zero_filled = zero_filled / scale
    label = label / scale
    zero_filled.requires_grad = True
    cg = CG_EEH().to(DEVICE)
    with torch.no_grad():
        us_ksp = cg(fft_loss(zero_filled, coil, cfg["axis"]) * om, coil, om)
        recon_0 = net(zero_filled, coil, om)
        Ex1 = fft_loss(recon_0, coil, cfg["axis"]) * om
        jitt = noise_jiterring(us_ksp, Ex1, om, std_scale=cfg["Mitigation"]["noise_jittering_std"]).to(DEVICE)
    return zero_filled, label, jitt


def make_attack(zero_filled, net, om, coil, cfg, eps):
    c = {**cfg, "Attack": {**cfg["Attack"], "epsilon": eps}}
    return attack_generation(zero_filled, net, DEVICE, om, coil, c)


def to_batch(item):
    ksp, coil, name = item
    return ksp.to(DEVICE), coil.to(DEVICE), name


def log(*a):
    print(*a, flush=True)
