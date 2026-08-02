#!/usr/bin/env python3
"""
score_run.py — post-run process-GT scorer for the Hermes-leaf + hy3:free control.

For one run directory (experiments/run03_hermes_leaf_pilot/<orid>/), this:

  1. Re-runs the agent's own `check_reproducibility.py` (if present) to CONFIRM
     the reported numbers are actually reproducible — not just asserted.
  2. Reads the agent's structured self-report `_run_meta.json` (verdicts,
     mutation-test flags, evidence-boundary flag, seeds, wall time, human
     interventions).
  3. Computes the design-doc primary outcomes for THIS control arm:
       execution_success, claim_coverage, full_evidence_rate (verified/falsified
       WITH a mutation test), inconclusive_rate, toy_rate, n_mutation_tests,
       has_evidence_boundary, wall_time_min, human_interventions,
       check_reproducibility_pass.
  4. Emits <dir>/_score.json and a short human-readable summary to stdout.

This is the "what did the single coding agent produce by itself" baseline
(RQ1 baseline, RQ5 evidence/time/compute/intervention). Comparison to the
silver reference (process-GT Level A) is a separate, later step.

Defensive: degrades gracefully if the agent deviated from the contract, and
flags exactly which fields are missing.
"""
import json, os, subprocess, sys, datetime

VERDICT_POINTS = {"verified": 2, "falsified": 2, "toy": 1, "inconclusive": 0}

def load_meta(d):
    p = os.path.join(d, "_run_meta.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception as e:
            return {"_parse_error": str(e)}
    return None

def run_check(d):
    """Return (passed_bool_or_None, detail_str). None = no checker present."""
    ck = os.path.join(d, "check_reproducibility.py")
    if not os.path.exists(ck):
        return None, "no check_reproducibility.py"
    # find a python: prefer .venv
    py = os.path.join(d, ".venv", "bin", "python")
    if not os.path.exists(py):
        py = sys.executable
    try:
        r = subprocess.run([py, "check_reproducibility.py"], cwd=d,
                           capture_output=True, text=True, timeout=1200)
        return (r.returncode == 0), (r.stdout + r.stderr)[-600:]
    except Exception as e:
        return False, f"checker error: {e}"

def score(d):
    meta = load_meta(d)
    bundle = json.load(open(os.path.join(d, "input_bundle.json")))
    n_anchored = len(bundle.get("anchored_claims", []))
    pc, detail = run_check(d)

    out = {
        "orid": bundle.get("orid"),
        "title": bundle.get("title"),
        "area": (json.load(open(os.path.join(d, "_meta.json"))).get("area")
                 if os.path.exists(os.path.join(d, "_meta.json")) else None),
        "n_anchored": n_anchored,
        "check_reproducibility_pass": pc,
        "check_detail": detail if pc is not True else "passed",
        "execution_success": None,
        "claim_coverage": None,
        "full_evidence_rate": None,
        "inconclusive_rate": None,
        "toy_rate": None,
        "n_mutation_tests": None,
        "has_evidence_boundary": None,
        "wall_time_min": None,
        "human_interventions": None,
        "per_claim_verdict": None,
        "warnings": [],
    }
    if not meta:
        out["warnings"].append("no _run_meta.json — agent deviated from contract")
        return out
    if meta.get("_parse_error"):
        out["warnings"].append(f"_run_meta.json parse error: {meta['_parse_error']}")
        return out

    pcv = meta.get("per_claim_verdict") or {}
    mut = meta.get("has_mutation_test") or {}
    out["execution_success"] = meta.get("execution_success")
    out["has_evidence_boundary"] = meta.get("has_evidence_boundary")
    out["human_interventions"] = meta.get("human_interventions")
    out["per_claim_verdict"] = pcv

    attempted = [k for k, v in pcv.items() if v not in (None, "not_attempted")]
    out["claim_coverage"] = round(len(attempted) / n_anchored, 3) if n_anchored else None

    full = sum(1 for k, v in pcv.items()
               if v in ("verified", "falsified") and mut.get(k) is True)
    incon = sum(1 for v in pcv.values() if v == "inconclusive")
    toy = sum(1 for v in pcv.values() if v == "toy")
    out["full_evidence_rate"] = round(full / n_anchored, 3) if n_anchored else None
    out["inconclusive_rate"] = round(incon / n_anchored, 3) if n_anchored else None
    out["toy_rate"] = round(toy / n_anchored, 3) if n_anchored else None
    out["n_mutation_tests"] = sum(1 for k, v in mut.items() if v is True)

    # wall time
    try:
        s = datetime.datetime.fromisoformat(meta["wall_time_start"])
        e = datetime.datetime.fromisoformat(meta["wall_time_end"])
        out["wall_time_min"] = round((e - s).total_seconds() / 60, 1)
    except Exception:
        out["warnings"].append("wall_time_start/end missing or unparseable")

    if meta.get("has_evidence_boundary") is not True:
        out["warnings"].append("no evidence boundary")
    if out["execution_success"] is not True:
        out["warnings"].append("execution_success != True")
    return out

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    d = os.path.abspath(d)
    out = score(d)
    with open(os.path.join(d, "_score.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    # human summary
    print(f"== score: {out['orid']} — {out.get('title','')[:50]}")
    print(f"   area={out.get('area')}  anchored={out['n_anchored']}")
    print(f"   check_reproducibility_pass = {out['check_reproducibility_pass']}")
    print(f"   execution_success          = {out['execution_success']}")
    print(f"   claim_coverage             = {out['claim_coverage']}")
    print(f"   full_evidence_rate         = {out['full_evidence_rate']}")
    print(f"   inconclusive_rate          = {out['inconclusive_rate']}")
    print(f"   toy_rate                   = {out['toy_rate']}")
    print(f"   n_mutation_tests           = {out['n_mutation_tests']}")
    print(f"   has_evidence_boundary      = {out['has_evidence_boundary']}")
    print(f"   wall_time_min              = {out['wall_time_min']}")
    print(f"   human_interventions        = {out['human_interventions']}")
    if out["warnings"]:
        print("   WARNINGS: " + " | ".join(out["warnings"]))

if __name__ == "__main__":
    main()
