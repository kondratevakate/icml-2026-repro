"""C00996 (mechanism): mitigation = reverse PGD inside an eps-ball minimising the cyclic measurement-consistency
loss averaged over several undersampling patterns.

Pre-registered in checks/plan.json (parameters fixed before the run: kx=64, ky=32, nc=15, M=3 cases, eps=0.01,
main run T=20 iterations, direction controls T_c=6 iterations, Config.yaml defaults otherwise).
The repository's Cyclic_Mitigation / ABA_detect / Reverse_PGD_Projection / CG_EEH run unchanged; two recording
hooks wrap ABA_detect and Reverse_PGD_Projection (they call the originals with identical arguments; the ascent
control only flips the sign of the gradient handed to Reverse_PGD_Projection).
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/checks")
from _common import (DATA, DEVICE, SynthDL, U, CG_EEH, IFFT, fft_loss, load_config, load_masks, load_network,
                     make_attack, param_hash, prepare_case, seed_everything, log)

DRY = bool(os.environ.get("P9_DRY"))   # wiring test only: 1 case, 2 iterations; output not used
KX, KY, NC = 64, 32, 15
M = 1 if DRY else 3
T = 2 if DRY else 20
TC = 2 if DRY else 6
EPS = 0.01
K_IDX = min(5, TC - 1)                  # compare LOSS[K_IDX] (after K_IDX updates) with LOSS[0]


def lines(mask):
    return frozenset(np.nonzero(mask.detach().cpu().numpy().reshape(mask.shape[-2], mask.shape[-1])[:, 0])[0].tolist())


def run_cyc(cfg_run, net, z_center, label, coil, om, ds, jitt, sign=1.0):
    rec = {"disp": [], "passes": [], "delta_lines": [], "omega_lines": None}
    st = {"passes": 0, "masks": []}
    orig_aba, orig_rev = U.ABA_detect, U.Reverse_PGD_Projection

    def aba(zf_temp, coil, Masks, model, config, jitt):
        st["passes"] += 1
        st["masks"].append(lines(Masks[1]))
        rec["omega_lines"] = lines(Masks[0])
        return orig_aba(zf_temp, coil, Masks, model, config, jitt)

    def rev(zf, zf_p, Grad, config, alpha):
        out = orig_rev(zf, zf_p, sign * Grad, config, alpha)
        d = out.detach() - zf_p.detach()
        rec["disp"].append(float(max(d.real.abs().max(), d.imag.abs().max())))
        rec["passes"].append(st["passes"])
        rec["delta_lines"].append(list(st["masks"]))
        st["passes"], st["masks"] = 0, []
        return out

    U.ABA_detect, U.Reverse_PGD_Projection = aba, rev
    try:
        PSNR, SSIM, LOSS = U.Cyclic_Mitigation(cfg_run, net, z_center, label.clone(), coil, om, ds, DEVICE, jitt)
    finally:
        U.ABA_detect, U.Reverse_PGD_Projection = orig_aba, orig_rev
    rec["LOSS"], rec["PSNR"], rec["SSIM"] = [float(x) for x in LOSS], [float(x) for x in PSNR], [float(x) for x in SSIM]
    return rec


def worker(i):
    torch.set_num_threads(1)
    t0 = time.time()
    om, ds = load_masks()
    net = load_network()
    h0 = param_hash(net)
    ksp, coil, _ = SynthDL(DATA + "/")[i]
    ksp, coil = torch.from_numpy(ksp).unsqueeze(0).to(DEVICE), torch.from_numpy(coil).unsqueeze(0).to(DEVICE)
    cfg_main = load_config(KX, KY, NC, iterations_mitigate=T)
    cfg_ctl = load_config(KX, KY, NC, iterations_mitigate=TC)
    cfg_one = load_config(KX, KY, NC, iterations_mitigate=1)
    seed_everything(100 + i)
    zf, label, jitt = prepare_case(ksp, coil, om, net, cfg_main)
    seed_everything(1000 * i + int(EPS * 1000))
    zp = make_attack(zf, net, om, coil, cfg_main, EPS)
    res = {"case": i, "attack_linf": float(max((zp - zf.detach()).real.abs().max(), (zp - zf.detach()).imag.abs().max()))}
    # supporting: (E_Omega^H)^dagger via the repo's CG, residual of E^H y = z
    with torch.no_grad():
        y = CG_EEH().to(DEVICE)(fft_loss(zp, coil, cfg_main["axis"]) * om, coil, om)
        Ey = torch.sum(IFFT(y * om) * torch.conj(coil), dim=1, keepdim=True)
        res["cg_pinv_residual"] = float(torch.linalg.norm(Ey - zp) / torch.linalg.norm(zp))
    res["clean_iter0"] = run_cyc(cfg_one, net, zf.detach(), label, coil, om, ds, jitt)      # LOSS[0] at the clean input
    res["main"] = run_cyc(cfg_main, net, zp, label, coil, om, ds, jitt, sign=+1.0)           # reverse PGD (descent)
    res["descent_ctl"] = run_cyc(cfg_ctl, net, zp, label, coil, om, ds, jitt, sign=+1.0)
    res["ascent_ctl"] = run_cyc(cfg_ctl, net, zp, label, coil, om, ds, jitt, sign=-1.0)      # sign-flipped control
    res["hash_before"], res["hash_after"] = h0, param_hash(net)
    res["seconds"] = time.time() - t0
    # sets are not JSON serialisable
    for k in ("clean_iter0", "main", "descent_ctl", "ascent_ctl"):
        res[k]["omega_lines"] = sorted(res[k]["omega_lines"]) if res[k]["omega_lines"] is not None else None
        res[k]["delta_lines"] = [[sorted(s) for s in it] for it in res[k]["delta_lines"]]
    log(f"case {i} done in {res['seconds']:.0f}s")
    return res


if __name__ == "__main__":
    t = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(M) as pool:
        R = pool.map(worker, range(M), chunksize=1)
    log(f"all workers done in {time.time() - t:.0f}s")
    eps_proj = load_config(KX, KY, NC)["Mitigation"]["epsilon_proj"]
    per = []
    for r in R:
        m = r["main"]
        omega = set(m["omega_lines"])
        dl = m["delta_lines"]
        distinct = all(len({tuple(x) for x in it}) == len(it) for it in dl)
        not_omega = all(set(x) != omega for it in dl for x in it)
        has_comp = all(len(set(x) - omega) >= 1 for it in dl for x in it)
        L = m["LOSS"]
        per.append({
            "case": r["case"],
            "a_ball_max_disp": max(m["disp"]), "a_ball_ok": max(m["disp"]) <= eps_proj + 1e-6,
            "a_moves_ok": max(m["disp"]) > 0.5 * eps_proj,
            "b_passes_per_iter": sorted(set(m["passes"])), "b_distinct_ok": distinct, "b_not_omega_ok": not_omega,
            "b_has_complement_line_ok": has_comp, "b_K3_ok": set(m["passes"]) == {3},
            "c_loss0": L[0], "c_loss_min": min(L), "c_loss_last": L[-1],
            "c_ok": (min(L) <= 0.95 * L[0]) and (L[-1] < L[0]),
            "d_descent_loss_k": r["descent_ctl"]["LOSS"][K_IDX], "d_ascent_loss_k": r["ascent_ctl"]["LOSS"][K_IDX],
            "d_loss0_ctl": r["descent_ctl"]["LOSS"][0],
            "d_ok": (r["descent_ctl"]["LOSS"][K_IDX] < r["descent_ctl"]["LOSS"][0]) and
                    (r["ascent_ctl"]["LOSS"][K_IDX] > r["ascent_ctl"]["LOSS"][0]),
            "ascent_ball_max_disp": max(r["ascent_ctl"]["disp"]),
            "support_attacked_loss0": L[0], "support_clean_loss0": r["clean_iter0"]["LOSS"][0],
            "support_attacked_gt_clean": L[0] > r["clean_iter0"]["LOSS"][0],
            "support_cg_pinv_residual": r["cg_pinv_residual"],
            "support_psnr_first_last_max": [m["PSNR"][0], m["PSNR"][-1], max(m["PSNR"])],
            "params_unchanged": r["hash_before"] == r["hash_after"],
        })
    crit = {
        "a_ball": all(p["a_ball_ok"] and p["a_moves_ok"] for p in per),
        "b_multi_pattern": all(p["b_K3_ok"] and p["b_distinct_ok"] and p["b_not_omega_ok"] and p["b_has_complement_line_ok"] for p in per),
        "c_minimisation": all(p["c_ok"] for p in per),
        "d_direction": all(p["d_ok"] for p in per),
    }
    summary = {"M": M, "T": T, "T_ctl": TC, "K_IDX": K_IDX, "eps_proj": eps_proj, "per_case": per, "criteria": crit,
               "supporting_attacked_gt_clean_all": all(p["support_attacked_gt_clean"] for p in per),
               "outcome": "consistent" if all(crit.values()) else "inconsistent"}
    log(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "raw": R}, open("/work/work/logs/C00996_result%s.json" % ("_DRY" if DRY else ""), "w"), indent=1)
