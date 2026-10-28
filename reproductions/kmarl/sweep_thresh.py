"""Fig.2-style threshold sweep: the regime where truncation bias is the dominant
term (true ARL comparable to / exceeding sequence length). Spec: see kmarl_repro.py."""
import numpy as np
from kmarl_repro import run_gsr, true_arl_add, make_dataset, estimators

SEEDS = list(range(10))
N = 1000
MAXLEN = 200000

if __name__ == "__main__":
    for thr in [50.0, 500.0, 5000.0, 50000.0]:
        g = np.random.default_rng(999)
        tARL, tADD, _, _ = true_arl_add(g, thr, n=3000, max_len=MAXLEN)
        print("=" * 104)
        print(f"GSR threshold {thr:>8.0f}   true ARL = {tARL:8.2f}   true ADD = {tADD:6.2f}")
        for label, (lo, hi) in [("T=1000 const", (1000, 1000)),
                                ("T~U[100,1000]", (100, 1000)),
                                ("T~U[30,300]", (30, 300))]:
            for prev in [0.1, 0.5, 0.9]:
                rows = []
                for s in SEEDS:
                    r = np.random.default_rng(7000 + s)
                    tau, nu, T = make_dataset(r, N, thr, lo, hi, prev, hi)
                    e = estimators(tau, nu, T)
                    rows.append([e['KM_ARL'] - tARL, e['LB_ARL'] - tARL, e['NV_ARL'] - tARL])
                a = np.array(rows)
                win = np.mean(np.abs(a[:, 0]) < np.abs(a[:, 1]))
                print(f"  {label:>14} prev={prev:.1f} | "
                      f"KM {a[:,0].mean():+9.1f}+/-{a[:,0].std():6.1f} (rel {a[:,0].mean()/tARL:+.3f}) | "
                      f"LB {a[:,1].mean():+9.1f}+/-{a[:,1].std():6.1f} (rel {a[:,1].mean()/tARL:+.3f}) | "
                      f"NV {a[:,2].mean():+9.1f} (rel {a[:,2].mean()/tARL:+.3f}) | KM<LB {win*10:.0f}/10")
