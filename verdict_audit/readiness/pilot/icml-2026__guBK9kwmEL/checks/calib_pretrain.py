"""Calibration (hyper-parameter sanity) of the pre-training learning rate, AUG arm only, training loss only
(no retrieval/robustness metric, no arm contrast).  Criterion: training loss clearly below chance 5*ln(16)=13.86."""
import sys
import rovtl_common as rc
rc.init(reduced=True)
from rovtl_common import *
D = "/work/work/data/tiny34x8"
prep = make_pretrain_prepper(D)
for lr in [float(a) for a in sys.argv[1:]]:
    log(f"CALIB lr={lr}")
    pretrain(D, True, 3, 16, lr, 5, f"/tmp/calib_{lr}.pth", prep=prep, freeze_image_trunk=True)
