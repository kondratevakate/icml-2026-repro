import numpy as np
from nn_eoss import MLP, synthetic_data, train, make_optimizer

# harder problem so the landscape curvature exceeds the plateaus
def hard_data(n=800, din=2, seed=0, scale=4.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=scale, size=(n, din))
    Y = np.sin(3.0 * X[:, 0]) + np.cos(2.0 * X[:, 1]) + 0.5 * (X[:, 0] * X[:, 1]) + 0.05 * rng.normal(size=n)
    return X.astype(np.float64), Y.astype(np.float64).reshape(-1, 1)

print("plateaus: 2/eta=20  2(1-beta)/eta=2  2(1+beta)/eta=38  (eta=0.1; for SGDM use eta=0.01 -> eta_eff=0.1)")

X, Y = hard_data(n=800, seed=1, scale=4.0)
print("\n--- SGD (eta=0.1) across batch ---")
for bs in [4, 16, 64, 256]:
    net = MLP(din=2, dh=32, dout=1, seed=2, init_scale=0.4)
    r = train(net, X, Y, "sgd", 0.1, 0.0, bs, steps=4000, measure_every=400, n_mc=25, seed=3)
    bs_h = r['batch_sharpness']
    print(f"  SGD b={bs:4d}: BS end~{bs_h[-1]:.2f}  (mid {bs_h[len(bs_h)//2]:.2f})  loss={r['loss'][-1]:.3f}")

print("\n--- SGDM (eta=0.01,beta=0.9 -> eta_eff=0.1) across batch ---")
for bs in [4, 16, 64, 256]:
    net = MLP(din=2, dh=32, dout=1, seed=2, init_scale=0.4)
    r = train(net, X, Y, "sgdm", 0.01, 0.9, bs, steps=4000, measure_every=400, n_mc=25, seed=3)
    bs_h = r['batch_sharpness']
    print(f"  SGDM b={bs:4d}: BS end~{bs_h[-1]:.2f}  (mid {bs_h[len(bs_h)//2]:.2f})  loss={r['loss'][-1]:.3f}")

print("\n--- SGDN (eta=0.01,beta=0.9) across batch ---")
for bs in [4, 16, 64, 256]:
    net = MLP(din=2, dh=32, dout=1, seed=2, init_scale=0.4)
    r = train(net, X, Y, "sgdn", 0.01, 0.9, bs, steps=4000, measure_every=400, n_mc=25, seed=3)
    bs_h = r['batch_sharpness']
    print(f"  SGDN b={bs:4d}: BS end~{bs_h[-1]:.2f}  (mid {bs_h[len(bs_h)//2]:.2f})  loss={r['loss'][-1]:.3f}")
