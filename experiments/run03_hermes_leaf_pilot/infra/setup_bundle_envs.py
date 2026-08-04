#!/usr/bin/env python3
"""
setup_bundle_envs.py — give every run03 paper bundle a ready-to-use Python env.

Problem this fixes (user directive 2026-08-01):
  make_bundles.py wrote TASK.md saying "stdlib + numpy/scipy/sympy only". Agents obeyed
  literally and FELL OVER on claims that need torch/sklearn or a downloaded dataset,
  then wrote `inconclusive` on work that was trivially fixable (just install torch,
  just download the data). That is a harness defect, not an agent capability limit.

Fix:
  1. Symlink `.venv` in EVERY bundle to a single shared base venv that already has
     numpy, scipy, sympy, torch (CPU build), scikit-learn, lightning, pandas.
     (Symlink, not copy: 42 copies of torch would waste ~30GB.)
     A paper may already have its own real `.venv` (e.g. Vv4XRZDMM0 built one with
     numpy/scipy/sympy while running) — in that case we UPGRADE it in place by
     installing the heavy deps into it, so the agent's existing artifacts keep working.
  2. Drop a short `ENV.md` into each bundle documenting the env + that data download
     is allowed/expected.
  3. Append an environment section to TASK.md (idempotent: skip if already present).

Idempotent: safe to re-run. Network: pip only if a dep is missing; the base venv is
built once by the caller (or we build it here if missing).
"""
import os, json, subprocess, sys, shutil

REPO = "/home/kate/projects/02_academia/icml-2026-repro"
OUT = os.path.join(REPO, "experiments/run03_hermes_leaf_pilot")
BASE_VENV = "/tmp/run03_base_venv"  # shared base, built by install step
HEAVY = ["numpy", "scipy", "sympy", "torch", "scikit-learn", "lightning", "pandas"]

ENV_MD = """\
# Environment for this reproduction run

A Python 3.12 virtualenv is available at `.venv` (symlinked to the shared base env).
Activate: `source .venv/bin/activate`  — or run scripts with `.venv/bin/python`.

Pre-installed (CPU-only):
  numpy, scipy, sympy, torch (CPU build), scikit-learn, lightning, pandas

You MAY use all of the above. `pip install` is allowed if you need something else.

## Datasets
Datasets are NOT pre-bundled. Downloading them is PART OF THE TASK — do not mark a
claim `inconclusive` merely because a dataset is needed. Fetch into `./data/` via
`wget`/`curl`, `huggingface-cli`, `torchvision.datasets`, or the paper's official
script. Only refuse for data that is genuinely inaccessible (paywalled / restricted
approval you cannot obtain / private). Keep it small + CPU-only to fit the budget.

Do NOT request GPU. If the paper's method needs GPU-only scale, reproduce a small
CPU-feasible proxy and report it as such.
"""

TASK_ENV_SECTION = """\n\n## Environment & data (added by harness)
- A pre-built `.venv` is provided with numpy, scipy, sympy, **torch (CPU)**,
  scikit-learn, lightning, pandas. Use it — do NOT refuse claims just because they
  need deep-learning or sklearn.
- Downloading datasets (WILDS/FMoW, CIFAR, UCI, etc.) into `./data/` is EXPECTED,
  not optional. Only mark `inconclusive` for data reasons if the data is truly
  inaccessible. A claim reproducible after a download must be attempted.
"""


def venv_python(path):
    return os.path.join(path, "bin", "python")


def ensure_base():
    if os.path.isdir(os.path.join(BASE_VENV, "bin")):
        print(f"[base] already present: {BASE_VENV}")
        return True
    print(f"[base] building {BASE_VENV} ...")
    subprocess.run([sys.executable, "-m", "venv", BASE_VENV], check=True)
    py = venv_python(BASE_VENV)
    subprocess.run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    # torch CPU from its own index, the rest from PyPI (do NOT set --index-url globally)
    subprocess.run([py, "-m", "pip", "install", "--quiet", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    subprocess.run([py, "-m", "pip", "install", "--quiet", *HEAVY[4:]], check=True)
    return True


def ensure_deps_in(venv_dir):
    """Make sure the heavy deps are present in a given venv (used for in-place upgrade)."""
    py = venv_python(venv_dir)
    if not os.path.exists(py):
        return False
    for dep in HEAVY:
        try:
            subprocess.run([py, "-c", f"import {dep.replace('-','_')}"],
                           capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print(f"  installing {dep} into {venv_dir} ...")
            extra = ["--index-url", "https://download.pytorch.org/whl/cpu"] if dep == "torch" else []
            subprocess.run([py, "-m", "pip", "install", "--quiet", dep, *extra], check=True)
    return True


def main():
    ensure_base()
    bundles = [d for d in os.listdir(OUT)
               if os.path.isdir(os.path.join(OUT, d)) and not d.startswith((".", "_"))
               and os.path.exists(os.path.join(OUT, d, "input_bundle.json"))]
    print(f"[scan] {len(bundles)} paper bundles")

    n_symlink = n_upgrade = n_envmd = n_task = 0
    for o in sorted(bundles):
        d = os.path.join(OUT, o)
        venv_dir = os.path.join(d, ".venv")

        if os.path.islink(venv_dir):
            continue  # already linked
        if os.path.isdir(venv_dir):
            # agent built its own real venv -> upgrade in place (keep its artifacts)
            if ensure_deps_in(venv_dir):
                n_upgrade += 1
                print(f"  [upgrade] {o}: deps added to existing .venv")
        else:
            os.symlink(BASE_VENV, venv_dir)
            n_symlink += 1
            print(f"  [symlink] {o} -> base venv")

        # ENV.md
        env_md = os.path.join(d, "ENV.md")
        if not os.path.exists(env_md):
            with open(env_md, "w") as f:
                f.write(ENV_MD)
            n_envmd += 1

        # append env section to TASK.md (idempotent)
        task = os.path.join(d, "TASK.md")
        if os.path.exists(task):
            t = open(task).read()
            if "Environment & data (added by harness)" not in t:
                with open(task, "a") as f:
                    f.write(TASK_ENV_SECTION)
                n_task += 1

    print(f"\nDONE. symlinked={n_symlink} upgraded={n_upgrade} "
          f"ENV.md written={n_envmd} TASK.md patched={n_task}")


if __name__ == "__main__":
    main()
