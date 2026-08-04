"""Re-run every verify_claim<N>.py and assert the stored evidence is
byte-identical (seed-pinned determinism) and not stale.

Usage:  ./venv/bin/python check_reproducibility.py
"""
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "venv", "bin", "python")
CLAIMS = [1, 2, 3, 4, 5, 6]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


ok = True
report = {}
for n in CLAIMS:
    script = os.path.join(HERE, f"verify_claim{n}.py")
    res = os.path.join(HERE, "results", f"claim{n}.json")
    before = sha(res)
    stale = os.path.getmtime(res) < os.path.getmtime(script)
    r = subprocess.run([PY, script], cwd=HERE, capture_output=True)
    after = sha(res)
    entry = dict(exit_code=r.returncode, stale_before_rerun=stale,
                 sha_before=before, sha_after=after,
                 deterministic=before == after,
                 verdict=json.load(open(res)).get("verdict"))
    report[f"claim{n}"] = entry
    ok &= (r.returncode == 0 and entry["deterministic"] and not stale)

print(json.dumps(dict(all_ok=ok, claims=report), indent=2))
sys.exit(0 if ok else 1)
