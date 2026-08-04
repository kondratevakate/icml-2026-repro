#!/usr/bin/env python3
"""run_arm2.py — launch an arm2 (Hermes + K-Dense skills) reproduction run for one
paper, with context-compaction enabled (saves router tokens on 33+ papers).

What it does:
  1. sync_data.py  <paper>  <run_dir>/data   (datasets/checkpoints from D:)
  2. builds the delegate prompt (arm2 + compaction rule + paper-claim-reproduction skill)
  3. prints the prompt + context so the orchestrator can dispatch delegate_task

Context-compaction rule (added to prompt): after every ~6 tool turns the agent MUST
write COMPACTION.md summarizing progress + ALL executed numbers verbatim; subsequent
turns reference COMPACTION.md instead of the full transcript. This cuts token volume
on long runs while preserving artifacts.

Usage: python3 run_arm2.py <orid> <paper_name_on_D> [--run-dir <dir>]
"""
import os, sys, argparse, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/kate/.venvs/neurofm_py311/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orid")
    ap.add_argument("paper")  # folder name on D: drive
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    run_dir = a.run_dir or os.path.join(
        REPO, "experiments/run03_hermes_leaf_pilot", f"{a.orid}_arm2")
    os.makedirs(run_dir, exist_ok=True)

    # 1) sync data
    print(f"# step 1: sync data from D:/{a.paper}")
    r = subprocess.run([PY, os.path.join(REPO, "groundtruth/sync_data.py"),
                        a.paper, run_dir], capture_output=True, text=True)
    print(r.stdout.strip()[-500:])
    if r.returncode != 0:
        print("  (sync warning, continuing)", r.stderr[-300:])

    # 2) build prompt (v2.1: agent must SELF-REMOVE blockers doable on this machine)
    prompt = f"""This is arm2 (Hermes + K-Dense skill set) reproduction of ICML-2026 paper orid {a.orid}.
Working dir already contains TASK.md, input_bundle.json (with anchored_claims + arxiv/openreview
ids), references/, and (if the paper needs datasets) a local data/ folder synced from the
reference drive.

GOAL: reproduce the paper's anchored claims via executable verify scripts on CPU. Use the
`paper-claim-reproduction` skill (K-Dense method skill). Model: tencent/hy3:free via
localhost:8319/v1 (router proxies, no key needed). CPU-only, no torch unless the paper demands it.

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
"no access to paper body" / "impossible" as a bare excuse -- if it was fixable, you should have
fixed it.

A mathematical claim (bound, identity, theorem) MUST be verified for real on CPU (numerical check
or independent re-derivation). Do NOT mark it "toy" -- toy is only for empirical claims where a
reduced-scale CPU run is the honest best effort.

CONTEXT-COMPACTION (mandatory for token efficiency across the 33+ paper batch):
- After every ~6 tool turns, write COMPACTION.md in the run dir: a tight summary of what was
  done, what failed, and ALL executed numbers verbatim (MSE values, violation counts, p-values).
- In later turns, reference COMPACTION.md instead of re-reading the full transcript.
- NEVER summarize away raw numeric results — those must stay exact.

DELIVERABLES in {run_dir}/:
- verify_claimN.py (one per anchored claim) + results/claimN.json with real executed numbers
- logbook.md with per-claim verdicts (verified / falsified / inconclusive / toy) from REAL runs,
  each with a one-line reason
- _run_meta.json: {{"arm":"arm2","agent":"Hermes + K-Dense skills","model":"tencent/hy3:free via localhost:8319/v1"}}
After finishing, the orchestrator will run score_run.py + publish_local.py (compare-and-publish to HF Space).
"""
    ctx = f"Run dir: {run_dir}\nPaper on D: /mnt/d/projects/02_academia/icml-repro/{a.paper}\n"
    print("\n" + "=" * 60)
    print("DISPATCH THIS TO delegate_task (role=leaf, background):")
    print("=" * 60)
    print("GOAL:", prompt)
    print("CONTEXT:", ctx)
    print("=" * 60)
    print(f"# to actually dispatch, the orchestrator calls delegate_task with the above.")


if __name__ == "__main__":
    main()
