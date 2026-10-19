"""C00096 - SleepBench size / sources (partial, metadata-only check; see checks/plan.json).

Uses ONLY the split-ID lists shipped in the repository (osf/splits/*.csv). No recording, no download.
Checks: total number of unique study IDs (claim: 'over 21,000'), ID families (claim: nine NSRR sources),
disjointness of the six splits. The '166,500 hours' figure cannot be checked from ID lists.
"""
import glob, os, re
from collections import Counter
import pandas as pd

NAMED = {"shhs": "SHHS", "cfs": "CFS", "ccshs": "CCSHS", "wsc": "WSC", "mros": "MROS", "mesa": "MESA", "chat": "CHAT", "sof": "SOF"}
CLAIM_SOURCES = set(NAMED.values()) | {"NCHSDB"}
files = sorted(glob.glob("work/repo/osf/splits/patient_*_ids.csv"))
sets, total_rows = {}, 0
for f in files:
    d = pd.read_csv(f)
    ids = d["patient_id"].astype(str)
    sets[os.path.basename(f)] = set(ids)
    total_rows += len(ids)
    print(f"{os.path.basename(f)}: rows={len(ids)} unique={ids.nunique()}")
union = set().union(*sets.values())
print("total rows:", total_rows, "| union of unique IDs:", len(union))
overlaps = [(a, b, len(sets[a] & sets[b])) for i, a in enumerate(sets) for b in list(sets)[i + 1:] if sets[a] & sets[b]]
print("pairwise split overlaps (non-empty):", overlaps if overlaps else "none (splits disjoint)")


def family(pid):
    m = re.match(r"^([A-Za-z]+)", pid)
    return m.group(1).lower() if m else "numeric"


fam = Counter(family(p) for p in union)
print("ID families (count of unique IDs):", dict(fam.most_common()))
named_found = {k for k in fam if k in NAMED}
unknown = {k for k in fam if k not in NAMED and k != "numeric"}
print("named-prefix families matching the claim's sources:", sorted(NAMED[k] for k in named_found))
print("families NOT among the claim's named sources (excluding the numeric family):", sorted(unknown) or "none")
print("numeric-pattern family present:", "numeric" in fam, "(assumed, not verified, to be NCHSDB; ID shape only)")
n_fam = len(fam)
print("number of ID families:", n_fam)
print("per-split family counts:")
for name, s in sets.items():
    print("   ", name, dict(Counter(family(p) for p in s).most_common()))
N = len(union)
print(f"implied mean hours per study if 166,500 h were spread over {N} IDs (arithmetic only, NOT a check): {166500 / N:.2f}")

ok_count = 21000 < N <= 22000
ok_fam = n_fam <= 9 and not unknown
print("RULE: 21,000 < N <= 22,000:", ok_count, "| <=9 families and none outside the named sources:", ok_fam)
outcome = "consistent" if (ok_count and ok_fam) else "inconsistent"
print("OUTCOME (pre-registered rule; hours figure NOT verified):", outcome)
