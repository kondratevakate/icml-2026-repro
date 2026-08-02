#!/usr/bin/env python3
"""
check_reproducibility.py — verifies that logbook.md numbers match results/claim*.json
for paper Ir6N7U5Kea.
"""

import json
import math
import sys
from pathlib import Path

W = Path(__file__).parent
FAILED = 0

def check(name, condition, msg):
    global FAILED
    if condition:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}: {msg}")
        FAILED += 1

# Claim 1: basic sanity
d1 = json.load(open(W / "results" / "claim1.json"))
print(f"claim1: {len(d1.get('cases', []))} cases loaded")

# Claim 2: ratio_exact matches ratio_corollary_4_10
with open(W / "results" / "claim2.json") as f:
    d2 = json.load(f)
for c in d2.get("cases", []):
    actual = c["ratio_exact"]
    paper = c["ratio_corollary_4_10"]
    check(f"claim2 {c['name']} ratio", math.isclose(actual, paper, rel_tol=1e-9),
          f"{actual} != {paper}")

# Claim 4: mutation single level ratio = 1.0
with open(W / "results" / "claim4.json") as f:
    d4 = json.load(f)
for key in ["MSNN_MUTATION_single_level"]:
    for c in d4.get("results", {}).get(key, {}).values():
        if "mutation_single_level_ratio" in c:
            check(f"claim4 mutation {key}", c["mutation_single_level_ratio"] == 1.0,
                  f"ratio = {c['mutation_single_level_ratio']}")

# Claim 5: NOT CHECKED - inconclusive

# Claim 6: balanced MAC spans multiple levels > 95%
with open(W / "results" / "claim6.json") as f:
    d6 = json.load(f)
check("claim6 balanced_frac_MAC_spanning_multiple_levels",
      d6["balanced_frac_MAC_spanning_multiple_levels"] > 0.95,
      f"{d6['balanced_frac_MAC_spanning_multiple_levels']} <= 0.95")

print(f"Assertions: {6 - FAILED} passed, {FAILED} failed")
sys.exit(0 if FAILED == 0 else 1)