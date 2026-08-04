#!/usr/bin/env python3
"""check_reproducibility.py — re-runnable gate for the arm2 reproduction of CAffNet
(OpenReview 20hdQQQrA4).

Default mode : re-assert every number quoted in logbook.md against results/claim*.json,
               plus the prose check (every quoted figure string must appear in logbook.md).
--rerun      : back up results/, re-execute every verify_claim*.py, require the JSON to
               reproduce bit-for-bit apart from `wall_seconds`, then run the assertions.
               Restores the backup on failure.
--negative-control : corrupt one stored number, prove the gate turns red, restore.

Exit code is non-zero iff anything fails.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
PY = str(ROOT / ".venv" / "bin" / "python")
SCRIPTS = [f"verify_claim{i}.py" for i in range(1, 6)]

PASS, FAIL = [], []


def chk(name, cond, actual=None):
    line = f"{'PASS' if cond else 'FAIL'} :: {name}" + (f"  [{actual}]" if actual is not None else "")
    (PASS if cond else FAIL).append(line)
    print(line)


def load(n):
    return json.loads((RES / f"claim{n}.json").read_text())


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def assertions():
    c1, c2, c3, c4, c5 = (load(i) for i in range(1, 6))

    # ---- claim 1 : Theorem 3.5 bound
    chk("c1 verdict verified", c1["verdict"] == "verified", c1["verdict"])
    chk("c1 instances == 2400", c1["n_instances"] == 2400, c1["n_instances"])
    chk("c1 no bound violations", c1["n_bound_violations"] == 0, c1["n_bound_violations"])
    chk("c1 CAffNet always feasible", c1["n_infeasible_caffnet"] == 0, c1["n_infeasible_caffnet"])
    chk("c1 branch coverage floor (Case-2 >= 500)", c1["n_case2_projected"] >= 500,
        c1["n_case2_projected"])
    chk("c1 max ratio < min bound (3+3*sqrt(3)=8.196)", c1["max_ratio_over_all"] < 8.196,
        c1["max_ratio_over_all"])
    chk("c1 mutation: HardNet-Aff infeasible on some instances",
        c1["mutation_hardnet_aff"]["n_infeasible"] > 0,
        c1["mutation_hardnet_aff"]["n_infeasible"])

    # ---- claim 2 : null-space term, four properties
    chk("c2 verdict verified", c2["verdict"] == "verified", c2["verdict"])
    chk("c2 (a) invariance < 1e-9", c2["a_invariance_max_abs_residual"] < 1e-9,
        c2["a_invariance_max_abs_residual"])
    chk("c2 (b) reachability < 1e-9", c2["b_reachability_max_abs_error"] < 1e-9,
        c2["b_reachability_max_abs_error"])
    chk("c2 (c) w=0 == orthogonal projection exactly",
        c2["c_degenerate_w0_vs_orthogonal_max_abs_diff"] == 0.0,
        c2["c_degenerate_w0_vs_orthogonal_max_abs_diff"])
    chk("c2 (d) w beats fixed projection on >50% instances",
        c2["d_frac_instances_w_beats_fixed"] > 0.5, c2["d_frac_instances_w_beats_fixed"])
    chk("c2 mutation w=0 exact on every instance",
        c2["mutation_w_zero"]["n_exact_matches"] == c2["n_instances"],
        c2["mutation_w_zero"]["n_exact_matches"])

    # ---- claim 3 : rank deficiency + cardinality
    chk("c3 verdict verified", c3["verdict"] == "verified", c3["verdict"])
    chk("c3 zero CAffNet violations", c3["n_violations_caffnet"] == 0, c3["n_violations_caffnet"])
    chk("c3 degeneracy floor (rank-deficient cases >= 500)", c3["n_rank_deficient"] >= 500,
        c3["n_rank_deficient"])
    chk("c3 enumeration size floor (cases >= 700)", c3["n_cases"] >= 700, c3["n_cases"])
    chk("c3 mutation HardNet-Aff violates", c3["mutation_hardnet_aff"]["n_violations"] > 0,
        c3["mutation_hardnet_aff"]["n_violations"])
    chk("c3 kmax == min(m, n_out) for every grid cell",
        c3["cardinality"]["kmax_equals_min_m_nout"] is True)
    chk("c3 mutation truncated enumeration fails",
        c3["mutation_truncated_enumeration"]["n_failures"] > 0,
        c3["mutation_truncated_enumeration"]["n_failures"])

    # ---- claim 4 : piecewise benchmark (split verdict)
    chk("c4 structural verdict verified", c4["structural"]["verdict"] == "verified",
        c4["structural"]["verdict"])
    chk("c4 CAffNet zero violations on the test grid",
        c4["structural"]["caffnet_n_violations"] == 0,
        c4["structural"]["caffnet_n_violations"])
    chk("c4 magnitude verdict capped at toy", c4["magnitude"]["verdict"] == "toy",
        c4["magnitude"]["verdict"])
    chk("c4 paper arithmetic 1-0.0012/0.0045 == 73.33%",
        near(c4["magnitude"]["paper_internal_arithmetic"]["value_pct"], 73.333333, 1e-4),
        c4["magnitude"]["paper_internal_arithmetic"]["value_pct"])
    chk("c4 magnitude NOT reproduced (measured << 73.33)",
        c4["magnitude"]["measured_reduction_pct"] < 60.0,
        c4["magnitude"]["measured_reduction_pct"])
    chk("c4 scale_note names the paper/this-run deficit", "GPU" in c4["scale_note"])

    # ---- claim 5 : control (unfavourable findings pinned too)
    chk("c5 CAffNet collision-free in every regime",
        c5["caffnet_collision_free_in_all_regimes"] is True)
    for r in c5["regimes"]:
        chk(f"c5 [{r}] CAffNet 0 collisions", c5["regimes"][r]["caff"]["n_collisions"] == 0)
        chk(f"c5 [{r}] CAffNet constraint violation < 1e-9",
            c5["regimes"][r]["caff"]["max_violation"] < 1e-9,
            c5["regimes"][r]["caff"]["max_violation"])
    chk("c5 single-obstacle regime does NOT discriminate (recorded failed prediction)",
        c5["regimes"]["single_obstacle_loose_box"]["hardnet"]["n_collisions"] == 0
        and c5["regimes"]["single_obstacle_loose_box"]["soft"]["n_collisions"] == 0)
    chk("c5 HardNet violates the hard constraint in the tight regime",
        c5["baseline_max_constraint_violation"]["two_obstacles_tight_box"]["hardnet"] > 1e-3,
        c5["baseline_max_constraint_violation"]["two_obstacles_tight_box"]["hardnet"])
    chk("c5 verdict is one of the honest set",
        c5["verdict"].startswith(("verified", "inconclusive", "partial")), c5["verdict"])
    # unfavourable findings are pinned too, so a later edit cannot drop them
    chk("c5 UNFAVOURABLE: no baseline collided in any regime (recorded)",
        c5["discriminating_regimes"] == [], c5["discriminating_regimes"])
    chk("c5 sub_verdict baselines_actually_collide == inconclusive",
        c5["sub_verdicts"]["baselines_actually_collide"] == "inconclusive")
    chk("c5 soft arm max violation 1.364 (loose regime)",
        near(c5["baseline_max_constraint_violation"]["single_obstacle_loose_box"]["soft"],
             1.3636, 5e-3))
    chk("c4 UNFAVOURABLE: soft baseline violation count == 413",
        c4["structural"]["soft_baseline_n_violations"] == 413,
        c4["structural"]["soft_baseline_n_violations"])
    chk("c4 measured reduction 9.22%",
        near(c4["magnitude"]["measured_reduction_pct"], 9.2235, 5e-3),
        c4["magnitude"]["measured_reduction_pct"])
    chk("c1 max ratio 1.71", near(c1["max_ratio_over_all"], 1.7138, 5e-3))
    chk("c1 mutation infeasible count == 92", c1["mutation_hardnet_aff"]["n_infeasible"] == 92)
    chk("c3 hardnet violation rate 0.0809",
        near(c3["mutation_hardnet_aff"]["violation_rate"], 0.08085, 5e-5))
    chk("c3 truncation failure rate 0.2533",
        near(c3["mutation_truncated_enumeration"]["failure_rate"], 0.25333, 5e-5))
    chk("c2 median relative loss reduction 0.517",
        near(c2["d_median_relative_loss_reduction"], 0.5174, 5e-4))

    # ---- verdict dictionary cannot be flipped silently
    verdicts = {1: c1["verdict"], 2: c2["verdict"], 3: c3["verdict"]}
    chk("verdict dict for claims 1-3 unchanged",
        verdicts == {1: "verified", 2: "verified", 3: "verified"}, verdicts)

    # ---- prose check: every quoted figure must appear in logbook.md
    lb = (ROOT / "logbook.md").read_text()
    for tok in ["2400", "706", "1.7137897456544158", "600", "705", "57", "114", "73.33",
                "413", "9.22", "0.620", "1.364", "25.33", "0.517", "92",
                "Evidence boundary"]:
        chk(f"logbook mentions '{tok}'", tok in lb)
    for n in range(1, 6):
        chk(f"logbook has a verdict row for claim {n}", f"| {n} |" in lb)


def rerun():
    bak = ROOT / "_results_backup"
    if bak.exists():
        shutil.rmtree(bak)
    shutil.copytree(RES, bak)
    try:
        before = {n: load(n) for n in range(1, 6)}
        for s in SCRIPTS:
            r = subprocess.run([PY, s], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:])
                raise SystemExit(f"{s} failed on re-run")
        for n in range(1, 6):
            a, b = before[n], load(n)
            a.pop("wall_seconds", None)
            b.pop("wall_seconds", None)
            chk(f"claim{n}.json reproduces bit-for-bit (modulo wall_seconds)",
                json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    except BaseException:
        shutil.rmtree(RES)
        shutil.copytree(bak, RES)
        raise
    finally:
        shutil.rmtree(bak, ignore_errors=True)


def negative_control():
    p = RES / "claim1.json"
    orig = p.read_text()
    d = json.loads(orig)
    d["n_bound_violations"] = 7                      # corrupt one stored number
    p.write_text(json.dumps(d, indent=2))
    r = subprocess.run([PY, str(Path(__file__).name)], cwd=ROOT, capture_output=True, text=True)
    p.write_text(orig)
    ok = r.returncode != 0
    print(f"{'PASS' if ok else 'FAIL'} :: negative control — gate turns red on a corrupted number")
    return 0 if ok else 1


def main():
    if "--negative-control" in sys.argv:
        raise SystemExit(negative_control())
    if "--rerun" in sys.argv:
        rerun()
    assertions()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
