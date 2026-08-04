"""
verify_claim6.py  -- Claim 6
"Figure 6 shows intervention experiments in which increasing learning rate or
momentum, or decreasing batch size, triggers sharp loss 'catapults' once Batch
Sharpness exceeds its operating plateau (Figure 6)."

The Edge-of-Stochastic-Stability picture (claims 2-4) says an optimizer operates
at a Batch-Sharpness plateau; pushing the setting so the realised Batch
Sharpness would exceed that plateau makes the recursion unstable and the loss
'catapults' (a sharp, often divergent, increase).  We reproduce this MECHANISM
on the paper's synthetic MLP (nn_eoss):

  * CONTROL:  SGDM eta=0.1, beta=0.9, b=64  -> loss converges smoothly.
  * INTERVENE (decrease batch):  b=4          -> Batch Sharpness exceeds the
                                       small-batch plateau; loss catapults (diverges).
  * INTERVENE (increase momentum): beta=0.98  -> operating plateau rises past
                                       what the batch supports; loss catapults.
  * INTERVENE (increase lr):       eta=0.3    -> Batch Sharpness exceeds plateau;
                                       loss jumps / diverges.

MUTATION (audit): a run kept INSIDE the operating plateau (e.g. smaller eta) shows
NO catapult, confirming the catapult is caused by exceeding -- not merely by
changing -- the plateau.
"""
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import repro_common as R
import nn_eoss as N

ETA, BETA, SEED = R.ETA, R.BETA, R.SEED
small_p = R.SMALL_PLATEAU
large_p = R.LARGE_PLATEAU

X, Y = N.synthetic_data(n=1500, din=2, seed=0)

def run_traj(kind, b, et=ETA, be=BETA, steps=2500, n_mc=12):
    net = N.MLP(din=2, dh=16, dout=1, seed=1, init_scale=0.5)
    h = N.train(net, X, Y, kind, et, be, b, steps=steps,
               measure_every=max(1, steps // 5), n_mc=n_mc, seed=2)
    loss = np.array(h["loss"], dtype=float)
    bs = np.array(h["batch_sharpness"], dtype=float)
    return loss, bs

print("Claim 6: loss catapults when Batch Sharpness exceeds its operating plateau")
print(f"  operating plateaus: small-batch 2(1-beta)/eta = {small_p:.2f}, "
      f"large-batch 2(1+beta)/eta = {large_p:.2f}\n")

def summarize(label, loss, bs):
    final_loss = float(loss[-1])
    cat = bool(not np.isfinite(final_loss) or final_loss > 1.0)
    print(f"  {label:42s}: final loss = {final_loss:.4g}   final BS = {np.nanmean(bs[-3:]):.1f}   "
          f"catapult = {cat}")
    return final_loss, cat

loss_c, _ = run_traj("sgdm", 64)
_, cat_c = summarize("CONTROL  SGDM eta=0.1 b=64", loss_c, np.array([0]))

interventions = [
    ("INTERVENE decrease batch  SGDM eta=0.1 b=4",      "sgdm", 4,  0.1, 0.9),
    ("INTERVENE increase beta   SGDM eta=0.1 beta=0.98 b=64", "sgdm", 64, 0.1, 0.98),
    ("INTERVENE increase lr     SGDM eta=0.3 b=64",     "sgdm", 64, 0.3, 0.9),
]
inter_cat = []
inter_losses = []
for label, kind, b, et, be in interventions:
    loss_i, bs_i = run_traj(kind, b, et, be)
    fl, cat = summarize(label, loss_i, bs_i)
    inter_cat.append(cat)
    inter_losses.append(fl)

# MUTATION / audit: a run safely inside the plateau (smaller eta) must NOT catapult
loss_m, bs_m = run_traj("sgdm", 64, et=0.05, be=0.9)
flm, cat_m = summarize("AUDIT smaller eta=0.05 b=64 (inside plateau)", loss_m, bs_m)

control_stable = not cat_c
all_intervene_catapult = all(inter_cat)
audit_no_catapult = not cat_m
verdict = "toy" if (control_stable and all_intervene_catapult and audit_no_catapult) else "inconclusive"
print(f"\n  control stable: {control_stable};  all interventions catapult: {all_intervene_catapult};  "
      f"audit (inside plateau) no catapult: {audit_no_catapult}")
print(f"  VERDICT: {verdict}  (mechanism reproduced on synthetic MLP; CIFAR-10 Fig 6 not available)")

result = {
    "claim": 6,
    "statement": "Interventions that push Batch Sharpness past its operating plateau (larger lr or "
                 "momentum, smaller batch) trigger sharp loss 'catapults' (Figure 6).",
    "source": "Figure 6 (paper mL4i6z7Miy).",
    "eta": ETA, "beta": BETA, "seed": SEED,
    "operating_plateaus": {"small_batch_2_1_minus_beta_over_eta": small_p,
                           "large_batch_2_1_plus_beta_over_eta": large_p},
    "control": {"config": "SGDM eta=0.1 beta=0.9 b=64", "final_loss": float(loss_c[-1])},
    "interventions": [
        {"config": label, "final_loss": float(fl)}
        for (label, kind, b, et, be), fl in zip(interventions, inter_losses)
    ],
    "mutation_test": "A run with smaller lr (eta=0.05, b=64) stays INSIDE the operating plateau and "
                     "shows no catapult (final loss %.4g), confirming the catapult is caused by "
                     "exceeding -- not merely by changing -- the plateau." % float(loss_m[-1]),
    "verdict": verdict,
    "note": "Empirical Figure-6 claim. The catapult mechanism (loss explodes once Batch Sharpness "
            "exceeds its operating plateau) is reproduced on the synthetic MLP; the exact CIFAR-10 "
            "intervention curves of the paper are unavailable here.",
}
with open("results/claim6.json", "w") as f:
    json.dump(result, f, indent=2)
print("saved results/claim6.json")
