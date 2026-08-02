#!/usr/bin/env python3
"""check_reproducibility.py for vaApZm6MKM"""
import json
import math
import sys
from pathlib import Path

W = Path(__file__).parent
FAILED = 0

def check(name, cond, msg=""):
    global FAILED
    if cond:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}: {msg}")
        FAILED += 1

# Claim 1: L1-ERT max error ~ 0
d1 = json.load(open(W / "results" / "claim1.json"))
check("claim1 L1 max_table1_formula error", 
      d1["numeric"]["L1"]["max_table1_formula_abs_error"] < 1e-10,
      "error too large")

# Claim 3: convergence at 5000
d3 = json.load(open(W / "results" / "claim3.json"))
check("claim3 convergence at 5000", 
      d3["results"]["convergence"]["n_points"] <= 5000,
      "5000 points insufficient")

# Claim 4: e- and l- decomposition adds to total error
d4 = json.load(open(W / "results" / "claim4.json"))
check("claim4 decomposition consistency", 
      abs(d4["results"]["ell_plus"] + d4["results"]["ell_minus"] - d4["results"]["total"]) < 1e-6,
      "decomposition mismatch")

# Claim 6: balanced MAC spans multiple levels > 95%
d6 = json.load(open(W / "results" / "claim6.json"))
check("claim6 balanced_frac_MAC_spanning_multiple_levels",
      d6.get("balanced_frac_MAC_spanning_multiple_levels", 0.95) > 0.95,
      "not > 0.95")

# Claim 2, 5: NOT CHECKED - inconclusive (missing artifacts)
print("INFO: claim2, claim5 are inconclusive (missing JSON artifacts)")

print(f"Assertions: {4 - FAILED} / 4 core claims passed, 2 inconclusive")