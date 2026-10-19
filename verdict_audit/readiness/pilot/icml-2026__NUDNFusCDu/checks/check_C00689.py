"""C00689 (mechanism): attack detection by zeta2 - zeta1 from cyclic re-undersampling, threshold, no retraining.

Pre-registered in checks/plan.json. Parameters (fixed before the run): kx=64, ky=32, nc=15, N=12 cases per
condition, eps in {0.01, 0.02}, Delta_1 for the primary statistic.
Uses the repository's frozen MoDL, attack_generation (unsupervised PGD), noise_jiterring, CG_EEH and ABA_detect.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/checks")
from _common import (DATA, DEVICE, SynthDL, U, CG_EEH, fft_loss, noise_jiterring, load_config, load_masks,
                     load_network, make_attack, param_hash, prepare_case, seed_everything, log)

DRY = bool(os.environ.get("P9_DRY"))   # dry run = wiring test only (2 cases, 1 attack iteration); output not used
KX, KY, NC, N = 64, 32, 15, (2 if DRY else 12)
EPS = [0.01, 0.02]


def zetas(z, coil, om, deltas, net, cfg, seed):
    """Defender view: only the (possibly perturbed) zero-filled image, coils, masks and the frozen network."""
    z = z.detach()
    out = {}
    with torch.no_grad():
        cg = CG_EEH().to(DEVICE)
        y = cg(fft_loss(z, coil, cfg["axis"]) * om, coil, om)            # y_Omega = (E^H)^dagger z (as us_ksp in Mitigation.py)
        rec0 = net(z, coil, om)
        Ex1 = fft_loss(rec0, coil, cfg["axis"]) * om
        torch.manual_seed(seed)
        jitt = noise_jiterring(y, Ex1, om, std_scale=cfg["Mitigation"]["noise_jittering_std"]).to(DEVICE)
        ny = torch.linalg.norm(y)
        for k, D in enumerate(deltas, 1):
            _, err = U.ABA_detect(z, coil, [om, D], net, cfg, jitt)          # stage 0 (Omega) and stage 1 (Delta_k)
            z1 = float(torch.linalg.norm(y - err[..., 0].to(DEVICE)) / ny)
            z2 = float(torch.linalg.norm(y - err[..., 1].to(DEVICE)) / ny)
            out[f"D{k}"] = {"z1": z1, "z2": z2, "s": z2 - z1}
        out["grad_enabled_during_detection"] = bool(torch.is_grad_enabled())
    return out


def psnr_of(net, z, coil, om, label):
    with torch.no_grad():
        r = net(z.detach(), coil, om)
    return float(U.getpsnr(np.abs(r.squeeze().numpy()), np.abs(label.detach().squeeze().numpy())))


def worker(i):
    torch.set_num_threads(1)
    cfg = load_config(KX, KY, NC)
    if DRY:
        cfg["Attack"]["iterations"] = 1
    om, ds = load_masks()
    net = load_network()
    h0 = param_hash(net)
    ksp, coil, _ = SynthDL(DATA + "/")[i]
    ksp, coil = torch.from_numpy(ksp).unsqueeze(0).to(DEVICE), torch.from_numpy(coil).unsqueeze(0).to(DEVICE)
    seed_everything(100 + i)
    t0 = time.time()
    zf, label, _ = prepare_case(ksp, coil, om, net, cfg)
    res = {"case": i}
    res["clean"] = zetas(zf, coil, om, ds, net, cfg, seed=7 + i)
    res["clean"]["psnr_recon"] = psnr_of(net, zf, coil, om, label)
    for eps in EPS:
        seed_everything(1000 * i + int(eps * 1000))
        zp = make_attack(zf, net, om, coil, cfg, eps)
        key = f"eps{eps}"
        res[key] = zetas(zp, coil, om, ds, net, cfg, seed=7 + i)
        res[key]["psnr_recon"] = psnr_of(net, zp, coil, om, label)
        res[key]["linf_pert"] = float(max((zp - zf.detach()).real.abs().max(), (zp - zf.detach()).imag.abs().max()))
    res["hash_before"], res["hash_after"] = h0, param_hash(net)
    res["seconds"] = time.time() - t0
    log(f"case {i} done in {res['seconds']:.0f}s")
    return res


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def best_threshold(pos, neg):
    cands = np.unique(np.concatenate([pos, neg]))
    best = (0.0, None)
    for tau in cands:
        tpr = np.mean(np.asarray(pos) >= tau)
        tnr = np.mean(np.asarray(neg) < tau)
        ba = 0.5 * (tpr + tnr)
        if ba > best[0]:
            best = (float(ba), float(tau))
    return best


if __name__ == "__main__":
    t = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        R = pool.map(worker, range(N), chunksize=1)
    log(f"all workers done in {time.time() - t:.0f}s")
    summary = {"N": N, "kx": KX, "ky": KY, "nc": NC}
    for D in ("D1", "D2", "D3"):
        clean_s = [r["clean"][D]["s"] for r in R]
        summary[D] = {"clean_s_median": float(np.median(clean_s)),
                      "clean_z1_median": float(np.median([r["clean"][D]["z1"] for r in R])),
                      "clean_z2_median": float(np.median([r["clean"][D]["z2"] for r in R]))}
        for eps in EPS:
            k = f"eps{eps}"
            att_s = [r[k][D]["s"] for r in R]
            ba, tau = best_threshold(att_s, clean_s)
            summary[D][k] = {"s_median": float(np.median(att_s)),
                             "z1_median": float(np.median([r[k][D]["z1"] for r in R])),
                             "z2_median": float(np.median([r[k][D]["z2"] for r in R])),
                             "auroc": auroc(att_s, clean_s), "best_balanced_acc": ba, "tau": tau,
                             "paired_frac_att_gt_clean": float(np.mean(np.asarray(att_s) > np.asarray(clean_s)))}
    summary["psnr_recon_median"] = {"clean": float(np.median([r["clean"]["psnr_recon"] for r in R]))}
    for eps in EPS:
        summary["psnr_recon_median"][f"eps{eps}"] = float(np.median([r[f"eps{eps}"]["psnr_recon"] for r in R]))
        summary[f"linf_pert_max_eps{eps}"] = float(max(r[f"eps{eps}"]["linf_pert"] for r in R))
    # criteria pre-registered in plan.json (primary statistic: Delta_1)
    S = summary["D1"]
    crit = {
        "i_auroc_ge_0.90_both_eps": all(S[f"eps{e}"]["auroc"] >= 0.90 for e in EPS),
        "ii_balanced_acc_ge_0.90_both_eps": all(S[f"eps{e}"]["best_balanced_acc"] >= 0.90 for e in EPS),
        "iii_median_z2_gt_median_z1_attacked": all(S[f"eps{e}"]["z2_median"] > S[f"eps{e}"]["z1_median"] for e in EPS),
        "iv_params_unchanged_and_no_grad": all(r["hash_before"] == r["hash_after"] for r in R)
        and all((not r["clean"]["grad_enabled_during_detection"]) and
                all(not r[f"eps{e}"]["grad_enabled_during_detection"] for e in EPS) for r in R),
    }
    summary["criteria"] = crit
    summary["outcome"] = "consistent" if all(crit.values()) else "inconsistent"
    log(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "per_case": R}, open("/work/work/logs/C00689_result%s.json" % ("_DRY" if DRY else ""), "w"), indent=1)
