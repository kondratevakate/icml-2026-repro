#!/usr/bin/env python3
"""run_batch.py — prepare arm2+compaction runs for a batch of papers (bundle 2).
For each orid: create run dir, copy TASK/input_bundle/references from the paper's base
folder (or a provided source), sync data from D:, and emit a delegate prompt file.

Does NOT dispatch — the orchestrator reads prompts/*.txt and calls delegate_task in
small parallel groups (<=3) to avoid router rate-limit.

Usage: python3 run_batch.py --papers TBSyYj4VV6 l35QweVxgn ... [--base-dir <dir-with-TASK>]
"""
import os, sys, argparse, json, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/kate/.venvs/neurofm_py311/bin/python"
COMMON = os.path.join(REPO, "experiments/run03_hermes_leaf_pilot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", required=True)
    ap.add_argument("--base-dir", default=COMMON,
                    help="dir containing <orid>/ with TASK.md etc (for copy)")
    a = ap.parse_args()
    out_prompts = os.path.join(COMMON, "batch_prompts")
    os.makedirs(out_prompts, exist_ok=True)

    for orid in a.papers:
        run_dir = os.path.join(COMMON, f"{orid}_arm2b")
        os.makedirs(run_dir, exist_ok=True)
        # copy base materials if present
        base = os.path.join(a.base_dir, orid)
        for f in ("TASK.md", "input_bundle.json"):
            src = os.path.join(base, f)
            if os.path.exists(src):
                subprocess.run(["cp", src, run_dir], check=False)
        for d in ("references", "paper"):
            src = os.path.join(base, d)
            if os.path.isdir(src):
                subprocess.run(["cp", "-r", src, run_dir], check=False)
        # venv symlink
        if not os.path.exists(os.path.join(run_dir, ".venv")):
            subprocess.run(["ln", "-s", "../20hdQQQrA4/.venv", os.path.join(run_dir, ".venv")], check=False)
        # sync data from D: (best-effort; many papers have none)
        subprocess.run([PY, os.path.join(REPO, "groundtruth/sync_data.py"),
                        orid, run_dir], capture_output=True, text=True)

        # write prompt (v2.1: agent must SELF-REMOVE blockers doable on this machine)
        prompt = f"""This is arm2 (Hermes + K-Dense skill set) reproduction of ICML-2026 paper orid {orid}.
Working dir: {run_dir} (has TASK.md, input_bundle.json with arxiv/openreview ids, paper/, venv).
GOAL: reproduce anchored claims via executable verify scripts on CPU. Use `paper-claim-reproduction` skill.
Model: tencent/hy3:free via localhost:8319/v1. CPU-only.

YOU HAVE THE POWER TO REMOVE MOST BLOCKERS YOURSELF -- do it before giving up:
- Missing paper text? Download it: arxiv id is in input_bundle.json -> fetch the PDF (e.g.
  https://arxiv.org/pdf/<arxiv_id>) into paper/ and read it. OpenReview may 403; try arxiv first.
- Missing library? `pip install` it into the venv yourself (numpy/scipy/sklearn/torch/etc).
- Empirical claim you cannot fully train? Design a TOY / reduced-scale CPU run that still
  exercises the claim's mechanism and report its (honest, real) numbers as evidence.
Only call a claim INCONCLUSIVE if it is TRULY impossible on this machine in a short time:
  * needs DATA that is absent AND cannot be fetched,
  * needs GPU-scale training that cannot run on CPU,
  * needs a library that genuinely cannot be installed.
Even then: state the SPECIFIC missing item, and still attempt any CPU-feasible proxy. Never write
"no access to paper body" / "impossible" as a bare excuse -- if it was fixable, you should have fixed it.

A mathematical claim (bound, identity, theorem) MUST be verified for real on CPU (numerical check
or independent re-derivation). Do NOT mark it "toy" -- toy is only for empirical claims where a
reduced-scale CPU run is the honest best effort.

CONTEXT-COMPACTION (mandatory): after every ~6 tool turns write COMPACTION.md (summary + ALL executed numbers verbatim). Later turns reference COMPACTION.md. Never summarize away raw numbers.
DELIVERABLES in {run_dir}/: verify_claimN.py + results/claimN.json (real numbers); logbook.md (per-claim verdicts with one-line reason each); _run_meta.json (arm=arm2, agent=Hermes+K-Dense skills, model=tencent/hy3:free via localhost:8319/v1).
"""
        ppath = os.path.join(out_prompts, f"{orid}.txt")
        open(ppath, "w").write(prompt)
        print(f"prepared {orid}: run_dir={run_dir} prompt={ppath}")

    print(f"\n{len(a.papers)} prompts written to {out_prompts}/")
    print("Dispatch via delegate_task in groups of <=3 (background).")


if __name__ == "__main__":
    main()
