import numpy as np
import cde_core as C

rng = C.seed_rng(0)
t = np.linspace(0, 3.0, 200)
Xclean = np.zeros((200, 2))
for j in range(2):
    Xclean[:, j] = np.sin(2 * np.pi * (0.7 + 0.3 * j) * t + rng.uniform(0, 6))
Xclean = Xclean / np.std(Xclean)
Xobs = Xclean + rng.normal(0.0, 0.25, size=Xclean.shape)

def run(method, pk=None):
    return C.cde_nfe(t, Xobs, method, tableau="DP45", tol=1e-4, vector_field="identity", path_kwargs=pk)["nfe"]

print("--- h-sweep (GP path over NOISY obs), expect NFE ~ h^-1 ---")
hs = np.array([0.05, 0.08, 0.12, 0.18, 0.28, 0.45, 0.7])
nfes = [run("gp", dict(h=float(hv), sigma2=1e-4)) for hv in hs]
for hv, n in zip(hs, nfes):
    print(f"  h={hv:.2f}  NFE={n}")
# fit only the power-law regime (exclude smooth plateau at large h)
reg = hs <= 0.28
sl_h = np.polyfit(np.log(hs[reg]), np.log(np.array(nfes)[reg]), 1)[0]
print("  slope (regime h<=0.28) =", round(sl_h, 3), "(target -1)")

print("\n--- method comparison (noisy obs, h=0.15) ---")
print(f"  linear  NFE={run('linear')}")
print(f"  cubic   NFE={run('cubic')}")
print(f"  gp      NFE={run('gp', dict(h=0.15, sigma2=1e-4))}")
print(f"  kernel  NFE={run('kernel', dict(h=0.15))}")

print("\n--- tol-sweep DP45 (p=5) on h=0.12 GP ---")
Xf, Xp = C.gp_smooth(t, Xobs, 0.12, sigma2=1e-4)
tols = np.array([1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6])
nfes2 = [C.adaptive_rk(lambda ti, z: Xp(ti), t[0], t[-1], np.zeros(2), tableau="DP45", tol=float(tv))["nfe"] for tv in tols]
for tv, n in zip(tols, nfes2):
    print(f"  tol={tv:.0e} NFE={n}")
sl_t = np.polyfit(np.log(tols), np.log(np.array(nfes2)), 1)[0]
print("  DP45 slope =", round(sl_t, 3), "(target -1/6 =", round(-1/6, 4), ")")

print("\n--- tol-sweep BS23 (p=3) ---")
nfes3 = [C.adaptive_rk(lambda ti, z: Xp(ti), t[0], t[-1], np.zeros(2), tableau="BS23", tol=float(tv))["nfe"] for tv in tols]
sl_t3 = np.polyfit(np.log(tols), np.log(np.array(nfes3)), 1)[0]
print("  BS23 slope =", round(sl_t3, 3), "(target -1/4 =", round(-0.25, 4), ")")
for tv, n in zip(tols, nfes3):
    print(f"    tol={tv:.0e} NFE={n}")
