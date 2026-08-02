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

        # write prompt
        prompt = f"""This is arm2 (Hermes + K-Dense skill set) reproduction of ICML-2026 paper orid {orid}.
Working dir: {run_dir} (has TASK.md, input_bundle.json, paper/, venv).
GOAL: reproduce anchored claims via executable verify scripts on CPU. Use `paper-claim-reproduction` skill.
Model: tencent/hy3:free via localhost:8319/v1. CPU-only.
CONTEXT-COMPACTION (mandatory): after every ~6 tool turns write COMPACTION.md (summary + ALL executed numbers verbatim). Later turns reference COMPACTION.md. Never summarize away raw numbers.
DELIVERABLES in {run_dir}/: verify_claimN.py + results/claimN.json (real numbers); logbook.md (per-claim verdicts); _run_meta.json (arm=arm2, agent=Hermes+K-Dense skills, model=tencent/hy3:free via localhost:8319/v1).
"""
        ppath = os.path.join(out_prompts, f"{orid}.txt")
        open(ppath, "w").write(prompt)
        print(f"prepared {orid}: run_dir={run_dir} prompt={ppath}")

    print(f"\n{len(a.papers)} prompts written to {out_prompts}/")
    print("Dispatch via delegate_task in groups of <=3 (background).")


if __name__ == "__main__":
    main()
