"""check_reproducibility.py -- reproducibility gate for this logbook.

Not a unit-test suite: it asserts that every number quoted in logbook.md is (a) still
produced by re-running the verify_claim*.py scripts and (b) still supports the verdict
that was assigned. Run it before trusting the logbook.

    ./.venv/bin/python check_reproducibility.py          # assertions only (fast)
    ./.venv/bin/python check_reproducibility.py --rerun  # also re-execute every script
                                                          # and diff against results/*.json

`--rerun` takes ~6 min (claim5 dominates). Exit code 0 iff everything holds.
"""
import json, subprocess, shutil, sys, pathlib

D = pathlib.Path(__file__).resolve().parent
PY = str(D / ".venv/bin/python")
R = D / "results"
SCRIPTS = {"verify_claim1.py": "claim1.json", "verify_claim1b.py": "claim1b.json",
           "verify_claim2.py": "claim2.json", "verify_claim3.py": "claim3.json",
           "verify_claim5.py": "claim5.json", "verify_claim5b.py": "claim5b.json"}
log, ok = [], True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    log.append(("PASS " if cond else "FAIL ") + msg)


def rerun(script, name):
    """Re-execute `script` and require it to reproduce results/`name` exactly."""
    stored = json.load(open(R / name))
    bak = R / (name + ".bak")
    shutil.copy(R / name, bak)
    p = subprocess.run([PY, script], cwd=D, capture_output=True, text=True, timeout=1800)
    if p.returncode:
        shutil.move(bak, R / name)
        return check(False, f"re-run {script} exited {p.returncode}: {p.stderr[-200:]}")
    strip = lambda d: {k: v for k, v in d.items() if k != "wall_seconds"}
    same = strip(stored) == strip(json.load(open(R / name)))
    bak.unlink()
    check(same, f"determinism: {script} reproduces {name} exactly")


def assertions():
    c1, c1b = json.load(open(R / "claim1.json")), json.load(open(R / "claim1b.json"))
    for a in ("0.05", "0.1", "0.2"):
        k, lvl = f"alpha={a}", float(a)
        check(c1["A_oracle"]["sign_test"][k] <= lvl,
              f"c1 oracle super-uniform @{a}: {c1['A_oracle']['sign_test'][k]:.4f}")
        check(c1["C_mutation_asymmetric_HRT_style_no_split"]["sign_test"][k] > lvl,
              f"c1 MUTATION inflates @{a}: "
              f"{c1['C_mutation_asymmetric_HRT_style_no_split']['sign_test'][k]:.4f}")
    check(all(r["oracle"] <= c1b["alpha"] for r in c1b["rows"]),
          f"c1b oracle <= alpha in all {len(c1b['rows'])} (n,p) configs")

    c2 = json.load(open(R / "claim2.json"))
    for q in ("0.05", "0.1", "0.2"):
        lvl = float(q)
        check(c2["MAIN_semi_knockoffs"][q]["FDR"] <= lvl,
              f"c2 FDR <= q={q}: {c2['MAIN_semi_knockoffs'][q]['FDR']:.4f}")
        for m in ("M1_mutation_no_plus_one_offset", "M2_mutation_asymmetric_statistic"):
            check(c2[m][q]["FDR"] > lvl, f"c2 {m[:2]} violates q={q}: {c2[m][q]['FDR']:.4f}")
    check(c2["MAIN_semi_knockoffs"]["0.2"]["power"] == 1.0, "c2 non-vacuous: power 1.0 @q=0.2")
    check(max(c2["M3_negative_control_rho_equals_nu"][q]["power"]
              for q in ("0.05", "0.1", "0.2")) < 0.05, "c2 M3 negative control: power collapses")

    c3 = json.load(open(R / "claim3.json"))
    t1, mu = c3["T1_rate_in_n_NULL_feature"], c3["MUTATION_non_null_feature"]
    check(t1["loglog_slope"] <= -0.45, f"c3 null slope {t1['loglog_slope']:.3f} <= -0.45")
    check(t1["R2"] > 0.95, f"c3 T1 fit R2 {t1['R2']:.3f}")
    check(abs(mu["loglog_slope"]) < 0.2, f"c3 MUTATION plateaus: {mu['loglog_slope']:.3f}")
    check(c3["ratio_nonnull_over_null_at_largest_n"] > 5,
          f"c3 mutation separated {c3['ratio_nonnull_over_null_at_largest_n']:.1f}x")
    check(all(v["R2"] > 0.85 for v in c3["T2_rate_in_delta_NULL_feature"].values()),
          "c3 T2 sqrt(log(1/delta)) fit R2 > 0.85 at every n")

    for n in (4, 6):
        j = json.load(open(R / f"claim{n}.json"))
        check(j["verdict"] == "inconclusive" and j.get("reason") and j.get("what_would_be_needed"),
              f"c{n} refusal is well-formed (verdict + reason + what-would-be-needed)")

    c5, c5b = json.load(open(R / "claim5.json")), json.load(open(R / "claim5b.json"))
    check(all(c5[s][m]["power"] == 1.0 for s in ("adjacent_support", "masked_correlation")
              for m in ("SKO-1", "SKO-5", "HRT")), "c5 ceiling effect real (why 5b exists)")
    for cfg, r in c5b["results"].items():
        check(0.0 < min(r[m]["power"] for m in r) and max(r[m]["power"] for m in r) < 1.0,
              f"c5b unsaturated [{cfg}]")
        check(r["SKO-5"]["type_I_error"] <= 0.05, f"c5b SKO-5 type-I ok [{cfg}]")
        check(r["SKO-5"]["power"] > r["SKO-1"]["power"], f"c5b derandomization helps [{cfg}]")
        # the logbook's honest caveat -- guarded so it cannot be quietly dropped
        check(r["SKO-1"]["power"] < r["HRT"]["power"], f"c5b caveat SKO-1 < HRT holds [{cfg}]")
    check(c5b["VERDICT_CAP"].startswith("toy"), "c5b self-labels as toy")


if __name__ == "__main__":
    if "--rerun" in sys.argv:
        for s, n in SCRIPTS.items():
            rerun(s, n)
    assertions()
    print("\n".join(log))
    print(f"\n{sum(l[0] == 'P' for l in log)} passed, {sum(l[0] == 'F' for l in log)} failed")
    sys.exit(0 if ok else 1)
