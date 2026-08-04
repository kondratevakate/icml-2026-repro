import numpy as np
from nn_eoss import MLP, synthetic_data, train

eta = 0.1
print(f"2/eta = {2/eta},  2(1-beta)/eta (beta=0.9) = {2*(1-0.9)/eta},  2(1+beta)/eta = {2*(1+0.9)/eta}")
X, Y = synthetic_data(n=600, din=2, seed=1)

for kind, beta, bs, tag in [("sgd", 0.0, 8, "SGD b=8"),
                             ("sgdm", 0.9, 8, "SGDM b=8 (small)"),
                             ("sgdm", 0.9, 256, "SGDM b=256 (large)")]:
    net = MLP(din=2, dh=24, dout=1, seed=2, init_scale=0.3)
    r = train(net, X, Y, kind, eta, beta, bs, steps=3000, measure_every=300, n_mc=25, seed=3)
    bs = r['batch_sharpness']
    print(f"{tag:18s}: BS start={bs[0]:.2f}  mid={bs[len(bs)//2]:.2f}  end={bs[-1]:.2f}  loss_end={r['loss'][-1]:.3f}")
