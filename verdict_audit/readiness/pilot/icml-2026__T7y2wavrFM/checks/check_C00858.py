"""Check for claim C00858 (mechanism): CAME-Grad is a backbone-agnostic optimizer with three
cascaded stages (S1 conflict-averse direction rectification, S2 magnitude-enhanced energy injection,
S3 adaptive gradient fusion). See checks/plan.json for the pre-registered sub-checks A-G and the
outcome rule. Implementation under test: checks/came_grad_from_paper.py (paper-derived; the
authors' optimizer is withheld). Synthetic data only.

usage: python checks/check_C00858.py [--smoke]
"""
import ast
import glob
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/checks")
from came_grad_from_paper import (CAME_Grad, came_direction, solve_dual_slsqp)  # noqa: E402
from synthetic_backbones import RRGNet, batches, load_split  # noqa: E402

SMOKE = "--smoke" in sys.argv
REPO = "/work/work/repo"
DATA = "/work/work/data"
OUTDIR = "/work/work/logs"
summary = {"smoke": SMOKE}
T0 = time.time()


def sec(t):
    print("\n=== " + t, flush=True)


# ---------------------------------------------------------------- A. repository audit
sec("A. Repository audit (static AST scan + import attempt)")
A = {}
A["CAME_Grad_file_present"] = os.path.exists(f"{REPO}/modules/CAME_Grad.py")
defs = []
for f in glob.glob(f"{REPO}/**/*.py", recursive=True):
    tree = ast.parse(open(f, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and "came" in node.name.lower():
            defs.append((os.path.relpath(f, REPO), node.name))
A["came_definitions_in_repo"] = defs
readme = open(f"{REPO}/README.md", encoding="utf-8").read()
A["readme_states_optimizer_withheld"] = "temporarily withheld" in readme
sys.path.insert(0, REPO)
try:
    importlib.import_module("modules.CAME_Grad")
    A["import_modules.CAME_Grad"] = "OK"
except Exception as e:  # noqa: BLE001
    A["import_modules.CAME_Grad"] = f"{type(e).__name__}: {e}"
tr = ast.parse(open(f"{REPO}/modules/trainer.py", encoding="utf-8").read())
calls = []
clip_in_branch = {"if_body": False, "else_body": False}
for node in ast.walk(tr):
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
        if name == "CAME_Grad":
            calls.append(("ctor", [ast.unparse(a) for a in node.args], [k.arg for k in node.keywords]))
        if name == "step" and isinstance(fn, ast.Attribute) and ast.unparse(fn).endswith("optimizer.step") \
                and (node.args or node.keywords):
            calls.append(("step", [ast.unparse(a) for a in node.args], [k.arg for k in node.keywords]))
    if isinstance(node, ast.If) and "use_came_grad" in ast.unparse(node.test) and "hasattr" in ast.unparse(node.test) \
            and any("task_weights" in ast.unparse(s) for s in node.body):
        clip_in_branch["if_body"] = any("clip_grad" in ast.unparse(s) for s in node.body)
        clip_in_branch["else_body"] = any("clip_grad" in ast.unparse(s) for s in node.orelse)
A["trainer_call_signatures"] = calls
A["trainer_grad_clip_value_by_branch"] = clip_in_branch
for k, v in A.items():
    print(f"  {k}: {v}")
# repo scheduler drives the wrapper's param_groups
from modules.optims import LinearWarmupCosineLRScheduler  # noqa: E402  (repository code)
_p = torch.nn.Parameter(torch.zeros(3))
_w = CAME_Grad(torch.optim.AdamW([{"params": [_p], "weight_decay": 0.05}], lr=1e-3))
_s = LinearWarmupCosineLRScheduler(_w, 8, 1e-4, 1e-3, decay_rate=None, warmup_start_lr=1e-5, warmup_steps=4)
_s.step(cur_epoch=1, cur_step=2)
lr_ep1 = _w.param_groups[0]["lr"]
_s.step(cur_epoch=4, cur_step=0)
lr_ep4 = _w.param_groups[0]["lr"]
A["repo_scheduler_sets_wrapper_lr"] = (abs(lr_ep1 - (1e-5 + (1e-3 - 1e-5) * 2 / 4)) < 1e-12 and lr_ep4 < 1e-3)
print("  repo LinearWarmupCosineLRScheduler drives wrapper.param_groups:", A["repo_scheduler_sets_wrapper_lr"],
      f"(lr epoch1/step2={lr_ep1:.3e}, epoch4={lr_ep4:.3e})")
summary["A"] = A

# ---------------------------------------------------------------- synthetic gradient sets
rng = np.random.default_rng(7)
KAPPA, NU = 1.5, 0.2
W_ALL = [1.0, 4.0, 1.0, 1.0]


def make_grads(rng, n, d, pattern, a):
    s = rng.standard_normal(d)
    s /= np.linalg.norm(s)
    if pattern == "aligned":
        lam = np.full(n, a)
    elif pattern == "alternating":
        lam = np.array([a if i % 2 == 0 else -a for i in range(n)])
    else:
        lam = rng.uniform(-a, a, size=n)
    N = rng.standard_normal((n, d))
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    r = np.exp(rng.uniform(math.log(0.3), math.log(3.0), size=n))
    return (r[:, None] * (lam[:, None] * s[None, :] + N)).astype(np.float64)


def neg_ratio(G):
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    C = Gn @ Gn.T
    iu = np.triu_indices(len(G), 1)
    return float((C[iu] < 0).mean())


def solve_primal(G, mu, xi):
    """Independent primal solve of Eq. 8 (epigraph form, SLSQP from the centre mu). The iterate is projected
    onto the trust-region ball so that its value min_i g_i^T u is a valid feasible (lower-bound) primal value,
    whatever the solver's success flag says."""
    from scipy.optimize import minimize
    n, d = G.shape
    x0 = np.concatenate([mu, [(G @ mu).min()]])
    cons = [{"type": "ineq", "fun": lambda x: G @ x[:d] - x[d], "jac": lambda x: np.hstack([G, -np.ones((n, 1))])},
            {"type": "ineq", "fun": lambda x: xi - np.sum((x[:d] - mu) ** 2),
             "jac": lambda x: np.concatenate([-2 * (x[:d] - mu), [0.0]])[None, :]}]
    res = minimize(lambda x: -x[d], x0, jac=lambda x: np.concatenate([np.zeros(d), [-1.0]]),
                   constraints=cons, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    u = res.x[:d]
    r = np.linalg.norm(u - mu)
    if r > math.sqrt(xi) and r > 0:
        u = mu + (u - mu) * (math.sqrt(xi) / r)
    return u, float((G @ u).min()), bool(res.success)


cases = []
reps = 1 if SMOKE else 2
for n in (2, 3, 4):
    for d in (5, 50):
        for pattern in ("aligned", "alternating", "random"):
            for a in (0.5, 1.0, 2.0, 4.0):
                for rho in (0.1, 0.2, 0.5, 0.9):
                    for _ in range(reps):
                        cases.append((n, d, pattern, a, rho, make_grads(rng, n, d, pattern, a)))
print(f"\nsynthetic gradient sets: {len(cases)} cases; mean negative-pairwise-cosine ratio "
      f"{np.mean([neg_ratio(c[5]) for c in cases]):.3f}; share of sets with >=1 conflict "
      f"{np.mean([neg_ratio(c[5]) > 0 for c in cases]):.3f}")

# ---------------------------------------------------------------- B, C, D, E, G1, G2
sec("B-E, G1-G2: stage-wise checks on synthetic task-gradient sets")
rec = {k: [] for k in ["dual_agree", "duality", "attain", "improve", "boundary", "s2_norm", "s2_dir",
                       "s3_interp", "s3_nu0", "s3_baseline", "primal_fail", "weak_duality"]}
abl = {"S1_off": [], "S2_off": [], "S3_off": []}
g1, g2 = [], []
worst = {}
for (n, d, pattern, a, rho, G) in cases:
    w = W_ALL[:n]
    Gt = torch.from_numpy(G)
    out = came_direction(Gt, w, rho, KAPPA, NU)
    mu, u, gj = out["mu"], out["u_rect"], out["g_joint"]
    c = float(rho * mu.norm())
    M, b = (Gt @ Gt.T), (Gt @ mu)
    Ft = out["dual"]
    _, Fs = solve_dual_slsqp(M.numpy(), b.numpy(), c)
    tolF = 1e-3 * max(1.0, abs(Ft))
    rec["dual_agree"].append(abs(Ft - Fs) <= tolF)
    _, P, ok = solve_primal(G, mu.numpy(), c * c)
    rec["duality"].append(abs(P - Ft) <= tolF)       # strong duality: feasible primal value reaches the dual value
    rec["weak_duality"].append(P <= Ft + tolF)       # sanity: no feasible primal value may exceed the dual value
    if not ok:
        rec["primal_fail"].append(True)
    mn_u = float((Gt @ u).min())
    rec["attain"].append(abs(mn_u - Ft) <= tolF)
    rec["improve"].append(mn_u >= float((Gt @ mu).min()) - 1e-9)
    rec["boundary"].append(abs(float((u - mu).norm()) - c) <= 1e-6 * c + 1e-12)
    # C: stage 2
    ue = out["u_en"]
    rec["s2_norm"].append(abs(float(ue.norm()) - KAPPA * float(gj.norm())) <= 1e-4 * KAPPA * float(gj.norm()))
    rec["s2_dir"].append(float(ue @ u / (ue.norm() * u.norm())) > 1 - 1e-9)
    # D: stage 3 (formula recomputed here from the stage outputs)
    exp = (1 - NU) * ue + NU * KAPPA * gj
    rec["s3_interp"].append(float((out["g_final"] - exp).norm()) <= 1e-9 * (1 + float(exp.norm())))
    o0 = came_direction(Gt, w, rho, KAPPA, 0.0)
    rec["s3_nu0"].append(float((o0["g_final"] - o0["u_en"]).norm()) <= 1e-9 * (1 + float(o0["u_en"].norm())))
    o1 = came_direction(Gt, w, rho, 1.0, 1.0)
    rec["s3_baseline"].append(float((o1["g_final"] - o1["g_joint"]).norm()) <= 1e-9 * (1 + float(o1["g_joint"].norm())))
    # E: stage separability (only on conflicting sets, rho in {0.2, 0.5})
    if neg_ratio(G) > 0 and rho in (0.2, 0.5):
        gf = out["g_final"]
        for name, st in (("S1_off", (False, True, True)), ("S2_off", (True, False, True)), ("S3_off", (True, True, False))):
            gv = came_direction(Gt, w, rho, KAPPA, NU, stages=st)["g_final"]
            abl[name].append(float((gf - gv).norm() / gf.norm()) > 1e-3)
    # G diagnostics
    g1.append((float(u.norm() / gj.norm()), neg_ratio(G) > 0))
    if rho == 0.5 and neg_ratio(G) > 0:
        g2.append((float((Gt @ gj).min()) < 0, mn_u < 0, float((Gt @ out["g_final"]).min()) < 0))


def frac(x):
    return float(np.mean(x)) if len(x) else float("nan")


for k in ["dual_agree", "duality", "weak_duality", "attain", "improve", "boundary", "s2_norm", "s2_dir", "s3_interp", "s3_nu0", "s3_baseline"]:
    print(f"  {k:12s}: pass {sum(rec[k])}/{len(rec[k])} = {frac(rec[k]):.4f}")
print(f"  primal SLSQP success flag False: {len(rec['primal_fail'])} cases (iterate still evaluated after projection)")
for k, v in abl.items():
    print(f"  ablation {k}: g_final differs from full pipeline (rel L2 > 1e-3) in {sum(v)}/{len(v)} = {frac(v):.4f}")
r = np.array([x[0] for x in g1])
rc = np.array([x[0] for x in g1 if x[1]])
print(f"  [G1 diagnostic] ||u_rect||/||g_joint||: median {np.median(r):.3f}, share <1: {np.mean(r < 1):.3f} "
      f"(conflicting sets only: median {np.median(rc):.3f}, share <1: {np.mean(rc < 1):.3f})")
g2a = np.array(g2)
print(f"  [G2 diagnostic] rho=0.5, conflicting sets, share with some task having negative inner product: "
      f"g_joint {g2a[:, 0].mean():.3f}, u_rect {g2a[:, 1].mean():.3f}, g_final {g2a[:, 2].mean():.3f}  (n={len(g2a)})")
summary["B_E"] = {k: [int(sum(v)), len(v)] for k, v in rec.items()}
summary["E_ablation"] = {k: [int(sum(v)), len(v)] for k, v in abl.items()}
summary["G1"] = {"median_ratio": float(np.median(r)), "share_lt1": float(np.mean(r < 1))}
summary["G2"] = {"g_joint": float(g2a[:, 0].mean()), "u_rect": float(g2a[:, 1].mean()), "g_final": float(g2a[:, 2].mean())}

# ---------------------------------------------------------------- F, D(e2e), G3: training on synthetic backbones
sec("F. Backbone-agnostic drop-in training runs (synthetic data, CPU)")
train_split = load_split(f"{DATA}/train.npz")
base_probs = np.array(json.load(open(f"{DATA}/base_probs.json")))
base_probs = base_probs / base_probs.max()


def train(kind, n_aux, arm, steps, seed=0, init_lr=1e-3, clip=False, base="adamw", kappa=KAPPA, nu=NU, rho=0.5,
          stages=(True, True, True), double=False, wd=0.05):
    torch.manual_seed(seed)
    model = RRGNet(kind, n_aux=n_aux)
    if double:
        model = model.double()
    p_wd, p_non = [], []
    for n_, p in model.named_parameters():
        (p_non if (p.ndim < 2 or "bias" in n_) else p_wd).append(p)
    groups = [{"params": p_wd, "weight_decay": wd}, {"params": p_non, "weight_decay": 0.0}]
    opt = (torch.optim.AdamW(groups, lr=init_lr, betas=(0.9, 0.999), weight_decay=wd) if base == "adamw"
           else torch.optim.SGD(groups, lr=init_lr))
    opt = CAME_Grad(opt, rho=rho, kappa=kappa, nu=nu, stages=stages) if arm == "came" else opt
    from modules.optims import LinearWarmupCosineLRScheduler
    sched = LinearWarmupCosineLRScheduler(opt, math.ceil(steps / 4), init_lr / 10, init_lr, decay_rate=None,
                                          warmup_start_lr=init_lr / 100, warmup_steps=4)
    th0 = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
    w = W_ALL[: n_aux + 1]
    hist, t0 = [], time.time()
    model.train()
    for i, batch in enumerate(batches(train_split, 8, steps)):
        sched.step(cur_epoch=i // 4 + 1, cur_step=i % 4)
        if double:
            batch = {k: (v.double() if v.is_floating_point() else v) for k, v in batch.items()}
        losses = model(batch, base_probs)
        if arm == "came":
            opt.step(losses, task_weights=w)
        else:
            loss = sum(wi * Li for wi, Li in zip(w, losses))
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            opt.step()
            opt.zero_grad()
        hist.append([float(L) for L in losses])
    th = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    return dict(hist=np.array(hist), theta=th, theta0=th0, secs=time.time() - t0,
                n_params=int(th.numel()))


STEPS = 8 if SMOKE else 32
Fres = []
for kind in ("mlp", "cnn", "transformer"):
    for K in (1, 2):
        row = {"backbone": kind, "n_aux_tasks": K}
        for arm in ("linear", "came"):
            try:
                o = train(kind, K, arm, STEPS)
                h = o["hist"]
                row[arm] = dict(ok=True, finite=bool(np.isfinite(h).all()),
                                lm_first4=float(h[:4, 0].mean()), lm_last4=float(h[-4:, 0].mean()),
                                lm_decreased=bool(h[-4:, 0].mean() < h[:4, 0].mean()),
                                secs=round(o["secs"], 1), n_params=o["n_params"])
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                row[arm] = dict(ok=False, error=f"{type(e).__name__}: {e}")
        Fres.append(row)
        print("  ", json.dumps(row), flush=True)
summary["F"] = Fres

sec("D (end-to-end, float64). nu=1, kappa=1 must reproduce the trainer's baseline path sum_i w_i L_i + AdamW (no clipping)")
Dres = []
for kind, K in (("mlp", 1), ("mlp", 2), ("cnn", 1), ("transformer", 1)):
    ob = train(kind, K, "linear", 10, clip=False, double=True)
    oc = train(kind, K, "came", 10, nu=1.0, kappa=1.0, double=True)
    diff = float((ob["theta"] - oc["theta"]).abs().max())
    moved = float((ob["theta"] - ob["theta0"]).abs().max())
    Dres.append(dict(backbone=kind, n_aux_tasks=K, max_abs_param_diff=diff, max_abs_param_move=moved, ok=diff < 1e-5))
    print("  ", json.dumps(Dres[-1]), flush=True)
summary["D_e2e"] = Dres

sec("G3 (diagnostic). Parameter displacement after 10 steps (weight decay 0), kappa=1.5 vs kappa=1.0, wrapped base optimizer AdamW vs SGD")
G3 = {}
for base, lr in (("adamw", 1e-3), ("sgd", 1e-2)):
    disp = {}
    for kappa in (1.0, 1.5):
        o = train("mlp", 1, "came", 10, base=base, init_lr=lr, kappa=kappa, wd=0.0)
        disp[kappa] = float((o["theta"] - o["theta0"]).norm())
    G3[base] = dict(disp_kappa1=disp[1.0], disp_kappa1_5=disp[1.5], ratio=disp[1.5] / disp[1.0])
    print(f"  {base}: ||theta_T-theta_0|| kappa=1.0: {disp[1.0]:.5f}  kappa=1.5: {disp[1.5]:.5f}  ratio {G3[base]['ratio']:.3f}")
summary["G3"] = G3

# ---------------------------------------------------------------- outcome by the pre-registered rule
sec("Outcome by pre-registered rule (plan.json, C00858)")
n_ok = lambda k: sum(rec[k]) / max(1, len(rec[k]))
cond = {}
cond["B"] = all(n_ok(k) >= 0.99 for k in ("dual_agree", "duality", "weak_duality", "attain", "improve", "boundary"))
cond["C"] = all(n_ok(k) >= 1.0 for k in ("s2_norm", "s2_dir"))
cond["D"] = (all(n_ok(k) >= 1.0 for k in ("s3_interp", "s3_nu0", "s3_baseline")) and all(x["ok"] for x in Dres))
cond["E"] = all(frac(v) >= 0.95 for v in abl.values())
cond["F"] = all(r_[a].get("ok") and r_[a].get("finite") and r_[a].get("lm_decreased") for r_ in Fres for a in ("came",))
for k, v in cond.items():
    print(f"  condition {k}: {'PASS' if v else 'FAIL'}")
outcome = "consistent" if all(cond.values()) else "inconsistent"
print("  OUTCOME:", outcome)
summary["conditions"] = cond
summary["outcome"] = outcome
summary["check_wall_seconds"] = round(time.time() - T0, 1)
if not SMOKE:
    json.dump(summary, open(f"{OUTDIR}/C00858_summary.json", "w"), indent=1, default=str)
print("wall seconds:", summary["check_wall_seconds"])
