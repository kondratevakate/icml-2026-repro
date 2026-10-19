"""C00080 - channel masking vs time masking vs both (paper Table 1, SimCLR rows 81.4 / 94.8 / 96.7 on real MROS).

SYNTHETIC mechanism check only (see checks/plan.json): the literal AUC numbers are NOT checked.
  Step A: fidelity of the repo's augmentation presets to the paper's description (Sec 3.2).
  Step B: SimCLR-style pre-training (NT-Xent; no SimCLR class in the repo, so written here) with the repo's
          vit_nano backbone and the repo's two-view augmentation registry, three presets x seeds,
          frozen-CLS linear probe, macro one-vs-rest AUC on held-out synthetic epochs.
Decision rule (pre-registered in plan.json) is applied at the end.
"""
import argparse, json, math, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/work/work/repo")
from osf.backbone.vit1d_cls import vit_nano
from osf.datasets.simclr_aug_registry import build_simclr_augmentor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PRESETS = {  # variant -> repo registry key
    "time_only": "pc_span_mask_aligned",        # per-channel span mask, ratio 0.3-0.6, no channel drop
    "channel_only": "channel_dropout_aligned",  # per-channel Bernoulli(0.5) drop, min_keep=1, no time mask
    "both": "chan_then_pcspan",                 # OSF recipe: channel drop 0.5 then span mask 0.3-0.6
}
GROUPS = {"other": [0, 1, 2, 3], "resp": [4, 5, 6, 7], "brain": [8, 9, 10, 11]}  # paper Sec 3.1 grouping
SCENARIOS = {  # paper Table 7 settings: which groups remain available at inference
    "headband(brain)": ["brain"],
    "resp+ecg": ["resp", "other"],
    "brain+ecg": ["brain", "other"],
    "resp_only": ["resp"],
}


# ------------------------------------------------------------------ Step A
def step_a(n=3000, seed=0):
    torch.manual_seed(seed)
    x = torch.ones(n, 12, 1920)
    out = {}
    for var, key in PRESETS.items():
        aug = build_simclr_augmentor(key)
        v1, v2 = aug(x)
        res = {}
        for tag, v in (("v1", v1), ("v2", v2)):
            zero = (v == 0)
            frac_t = zero.float().mean(-1)                       # [n,12] zero fraction per channel
            ch_dead = (frac_t == 1.0)                            # fully masked channels
            kept = ~ch_dead
            res[tag] = {
                "channels_fully_dropped_mean_frac": float(ch_dead.float().mean()),
                "min_channels_kept": int(kept.sum(1).min()),
                "mean_channels_kept": float(kept.sum(1).float().mean()),
                "span_zero_frac_in_kept_channels_min": float(frac_t[kept].min()),
                "span_zero_frac_in_kept_channels_max": float(frac_t[kept].max()),
                "span_zero_frac_in_kept_channels_mean": float(frac_t[kept].mean()),
                "kept_channels_with_zero_span_frac": float((frac_t[kept] > 0).float().mean()),
            }
        k1 = ((v1 == 0).float().mean(-1) < 1.0); k2 = ((v2 == 0).float().mean(-1) < 1.0)
        inter = (k1 & k2).sum(1).float(); union = (k1 | k2).sum(1).float()
        res["views_identical_channel_set_frac"] = float((k1 == k2).all(1).float().mean())
        res["mean_jaccard_of_kept_channel_sets"] = float((inter / union).mean())
        out[var] = res
        print(f"[A] {var:13s} ({key})")
        print(f"    v1: {json.dumps(res['v1'])}")
        print(f"    views with identical kept-channel set: {res['views_identical_channel_set_frac']:.4f}; "
              f"mean Jaccard of kept sets: {res['mean_jaccard_of_kept_channel_sets']:.3f}")
    return out


# ------------------------------------------------------------------ Step B
PATCH_T = 240   # patch length in samples (1920/240 = 8 time patches x 3 channel patches = 24 tokens)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_nano(num_leads=12, seq_len=1920, patch_size=PATCH_T, lead_wise=1, patch_size_ch=4)
        w = self.backbone.width
        self.proj = nn.Sequential(nn.Linear(w, 256), nn.GELU(), nn.Linear(256, 128))

    def embed(self, x):
        return self.backbone(x)   # CLS token after final LayerNorm (repo ViT.forward)


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    B = z1.shape[0]
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)])
    return F.cross_entropy(sim, tgt)


@torch.no_grad()
def embed_all(model, X, bs=250, keep_groups=None):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs])
        if keep_groups is not None:
            m = torch.zeros(1, 12, 1)
            for g in keep_groups:
                m[0, GROUPS[g], 0] = 1.0
            xb = xb * m
        outs.append(model.embed(xb).numpy())
    return np.concatenate(outs)


def probe_eval(model, Xtr, ytr, Xte, yte):
    Etr = embed_all(model, Xtr)
    sc = StandardScaler().fit(Etr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Etr), ytr)

    def auc(E):
        return float(roc_auc_score(yte, clf.predict_proba(sc.transform(E)), multi_class="ovr", average="macro"))
    res = {"full": auc(embed_all(model, Xte))}
    for name, groups in SCENARIOS.items():
        res[name] = auc(embed_all(model, Xte, keep_groups=groups))
    res["missing_mean"] = float(np.mean([res[k] for k in SCENARIOS]))
    return res


def train_one(variant, seed, steps, bs, lr, Xpre):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SimCLR()
    aug = build_simclr_augmentor(PRESETS[variant])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = np.random.default_rng(seed)
    n = len(Xpre)
    t0 = time.time(); losses = []
    model.train()
    for s in range(steps):
        idx = rng.choice(n, size=bs, replace=False)
        x = torch.from_numpy(Xpre[idx])
        v1, v2 = aug(x)
        loss = nt_xent(model.proj(model.backbone(v1)), model.proj(model.backbone(v2)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if (s + 1) % max(1, steps // 5) == 0 or s == 0:
            print(f"    [{variant} seed{seed}] step {s+1}/{steps} loss {np.mean(losses[-20:]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model, float(np.mean(losses[-20:])), time.time() - t0


def decide(res):
    v = ["time_only", "channel_only", "both"]
    m = {k: np.mean([r["full"] for r in res[k]]) for k in v}
    sd = {k: (np.std([r["full"] for r in res[k]], ddof=1) if len(res[k]) > 1 else float("nan")) for k in v}
    pooled = lambda a, b: math.sqrt((sd[a] ** 2 + sd[b] ** 2) / 2)
    info = {"mean_full_auc": m, "sd_full_auc": sd,
            "gap_both_minus_channel": m["both"] - m["channel_only"], "pooled_sd_both_channel": pooled("both", "channel_only"),
            "gap_channel_minus_time": m["channel_only"] - m["time_only"], "pooled_sd_channel_time": pooled("channel_only", "time_only")}
    if max(m.values()) < 0.60:
        return "not_checkable_on_synthetic", "best variant full-channel AUC < 0.60 (nothing learned)", info
    if m["both"] <= m["channel_only"] or m["channel_only"] <= m["time_only"]:
        return "inconsistent", "ordering both > channel_only > time_only contradicted in the means", info
    if info["gap_both_minus_channel"] > info["pooled_sd_both_channel"] and info["gap_channel_minus_time"] > info["pooled_sd_channel_time"]:
        return "consistent", "both > channel_only > time_only with every gap > pooled seed SD", info
    return "not_checkable_on_synthetic", "ordering holds in the means but a gap is within pooled seed SD", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--patch_t", type=int, default=240)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(PRESETS))
    ap.add_argument("--skip_a", action="store_true")
    ap.add_argument("--no_random_init", action="store_true")
    ap.add_argument("--out", default="work/logs/C00080_results.json")
    a = ap.parse_args()
    global PATCH_T
    PATCH_T = a.patch_t
    torch.set_num_threads(a.threads)
    print("args", vars(a), "torch", torch.__version__, "threads", torch.get_num_threads(), flush=True)
    result = {"args": vars(a)}
    if not a.skip_a:
        result["step_a"] = step_a()
    pre = np.load("work/data/pretrain.npz"); trn = np.load("work/data/probe_train.npz"); tst = np.load("work/data/test.npz")
    Xpre, Xtr, ytr, Xte, yte = pre["X"], trn["X"], trn["y"], tst["X"], tst["y"]
    res = {k: [] for k in a.variants}
    result["runs"] = {}
    if not a.no_random_init:
        ri = []
        for sd in a.seeds:
            torch.manual_seed(sd)
            m = SimCLR()
            ri.append(probe_eval(m, Xtr, ytr, Xte, yte))
        result["random_init"] = ri
        print("[B] random-init encoder full-channel AUC per seed:", [round(r["full"], 4) for r in ri],
              "| missing_mean:", [round(r["missing_mean"], 4) for r in ri], flush=True)
    for var in a.variants:
        for sd in a.seeds:
            model, last_loss, secs = train_one(var, sd, a.steps, a.bs, a.lr, Xpre)
            r = probe_eval(model, Xtr, ytr, Xte, yte)
            r.update(final_loss=last_loss, train_seconds=secs)
            res[var].append(r)
            result["runs"][f"{var}_seed{sd}"] = r
            print(f"[B] {var:13s} seed{sd}: full AUC {r['full']:.4f} | " +
                  " ".join(f"{k} {r[k]:.4f}" for k in SCENARIOS) + f" | missing_mean {r['missing_mean']:.4f} | final loss {last_loss:.3f} | {secs:.0f}s", flush=True)
            with open(a.out, "w") as f:
                json.dump(result, f, indent=1)
    if set(a.variants) == set(PRESETS) and len(a.seeds) > 1 and all(len(res[k]) > 0 for k in PRESETS):
        print("\n[B] summary (mean +- sd over seeds)")
        for k in PRESETS:
            fm = [r["full"] for r in res[k]]; mm = [r["missing_mean"] for r in res[k]]
            print(f"    {k:13s} full-channel AUC {np.mean(fm):.4f} +- {np.std(fm, ddof=1) if len(fm)>1 else float('nan'):.4f}"
                  f" | mean over 4 missing-channel scenarios {np.mean(mm):.4f} +- {np.std(mm, ddof=1) if len(mm)>1 else float('nan'):.4f}")
        outcome, why, info = decide(res)
        result["decision"] = {"outcome": outcome, "reason": why, "info": info}
        print("DECISION (pre-registered rule, primary metric = full-channel macro-AUC):", outcome, "-", why)
        print("    ", json.dumps(info))
        miss = {k: np.mean([r["missing_mean"] for r in res[k]]) for k in PRESETS}
        print("    secondary ordering on missing-channel mean AUC:", {k: round(v, 4) for k, v in miss.items()},
              "| both>=channel_only>time_only:", bool(miss["both"] >= miss["channel_only"] > miss["time_only"]))
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
