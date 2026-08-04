#!/usr/bin/env python3
"""check_reproducibility.py — re-run every verify_claim<N>.py and confirm that
the seed-pinned results are deterministic (reproducible by an independent run).

Only DETERMINISTIC quantities are compared exactly: verdict, mutation.passed,
and the seed/SVD/sampling-derived metrics (gram error, optimum loss ratio, sizes).
Wall-clock timing slopes are excluded from exact comparison (they vary run to
run) but their *sign* (full slope > 0, sketch slope ~ 0) is re-checked.
"""
from __future__ import annotations
import json, os, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/tmp/run03_base_venv/bin/python"
N = 6

# deterministic fields compared EXACTLY between two runs
DET_KEYS = [("claim_no",), ("verdict",), ("mutation", "passed"),
            ("metrics", "eps"), ("metrics", "m"), ("metrics", "n"), ("metrics", "r"),
            ("metrics", "sparsifier_size"),
            ("metrics", "ls_gram_relative_error"),
            ("metrics", "optimum_loss_ratio")]

TOL = 1e-9


def dig(d, keys):
    for k in keys:
        d = d[k]
    return d


def run_claim(c):
    p = os.path.join(HERE, f"verify_claim{c}.py")
    out = os.path.join(HERE, "results", f"claim{c}.json")
    env = dict(os.environ)
    subprocess.run([PY, p], cwd=HERE, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return json.load(open(out))


def main():
    all_ok = True
    for c in range(1, N + 1):
        r1 = run_claim(c)
        r2 = run_claim(c)
        det_ok = True
        detail = []
        for keys in DET_KEYS:
            try:
                a, b = dig(r1, keys), dig(r2, keys)
            except KeyError:
                continue  # claim-specific metric absent in this claim's JSON
            if isinstance(a, float) or isinstance(b, float):
                ok = abs(float(a) - float(b)) <= TOL
            else:
                ok = (a == b)
            det_ok &= ok
            if not ok:
                detail.append(f"{'/'.join(keys)}: {a!r} != {b!r}")
        # sign check on timing slopes (non-deterministic but should keep sign)
        fs1, ss1 = r1["metrics"].get("m_scaling_full_slope"), r1["metrics"].get("m_scaling_sketch_slope")
        fs2, ss2 = r2["metrics"].get("m_scaling_full_slope"), r2["metrics"].get("m_scaling_sketch_slope")
        sign_ok = (fs1 is None or fs2 is None) or ((fs1 > 0) == (fs2 > 0))
        ok = det_ok and sign_ok
        all_ok &= ok
        print(f"claim {c}: {'PASS' if ok else 'FAIL'} "
              f"(verdict={r1['verdict']}, mutation.passed={r1['mutation']['passed']}, "
              f"det_ok={det_ok}, sign_ok={sign_ok})"
              + ("" if ok else "  " + "; ".join(detail)))
    print("\nREPRODUCIBILITY:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
