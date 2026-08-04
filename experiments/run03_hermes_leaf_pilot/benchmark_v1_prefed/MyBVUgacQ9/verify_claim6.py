"""verify_claim6.py -- CLAIM 6 (Section 5 / Figures 5 and 6).

Claim: "The method is validated on 2D locomotion simulation and on real robot
experiments including a humanoid jump and a quadruped bounding task (Figures 5 and 6)."

This script does NOT fabricate a result.  It records, in machine-readable form, why the
real-robot half of the claim cannot be reproduced here, and points at the evidence that
IS available for the simulation half (claim 4's artifacts).

Run:  .venv/bin/python verify_claim6.py
"""
import json
import os
import time

OUT = "results/claim6.json"
CMD = ".venv/bin/python verify_claim6.py"


def main():
    t0 = time.time()
    res = {
        "claim": 6,
        "source": "Section 5, Figures 5 and 6 (also Figure 3)",
        "command": CMD,
        "verdict": "inconclusive",
        "parts": {
            "2D_locomotion_simulation": {
                "status": "partially covered by claim 4",
                "evidence": "results/claim4.json",
                "gap": ("The paper does not report the horizon T, the target CoM area, the "
                        "contact schedule, the contact locations r_i^j, the initial state "
                        "(c_init, cdot_init, k_init) or rho for Figure 5, so the *specific* "
                        "curves of Figure 5 cannot be reproduced; only the qualitative "
                        "mechanism (eq. 6, Corollary 4.2) is reproduced in claim 4 with our "
                        "own instance."),
            },
            "humanoid_vertical_jump_hardware": {
                "status": "not reproducible",
                "reasons": [
                    "requires physical humanoid hardware (Figure 6, top row)",
                    "no code, no URDF/robot model, no logs, no rosbags released",
                    "kinematics tracking uses an external DDP implementation (Crocoddyl, "
                    "Mastalli et al. 2020) with unreported cost weights and gains",
                    "Figure 3's dynamic-violation curves (10 randomized trials, 3 values of "
                    "Dt) come from that unreleased pipeline",
                ],
            },
            "quadruped_bounding_hardware": {
                "status": "not reproducible",
                "reasons": [
                    "requires physical quadruped hardware (Figure 6, bottom row)",
                    "same missing-code / missing-log / missing-parameters situation",
                ],
            },
        },
        "policy": ("Per the task rules, a hardware-only claim gets an honest 'inconclusive' "
                   "rather than a synthetic toy substituted for the real experiment. No "
                   "simulated jump/bound was run and presented as if it were the paper's "
                   "hardware result."),
        "claim4_artifact_present": os.path.exists("results/claim4.json"),
        "elapsed_sec": 0.0,
    }
    res["elapsed_sec"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("claim 6 verdict:", res["verdict"])
    for k, v in res["parts"].items():
        print(" -", k, ":", v["status"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
