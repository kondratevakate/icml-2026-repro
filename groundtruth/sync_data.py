#!/usr/bin/env python3
"""sync_data.py — copy a paper's datasets/checkpoints from the Windows D: drive
(/mnt/d/projects/02_academia/icml-repro/<paper>/) onto native ext4 before rerunning,
so the agent runs fast (9P mount is slow).

Usage: python3 sync_data.py <paper_name> <dest_run_dir> [--dry-run]
  <paper_name>  folder name under D:/projects/02_academia/icml-repro/ (e.g. calpro)
  <dest_run_dir> target run folder (e.g. experiments/run03_hermes_leaf_pilot/<orid>_arm2)
Copies: data/ + any *.ckpt *.pt *.pth *.h5 checkpoints found under the paper folder.
"""
import os, sys, shutil, argparse

SRC_ROOT = "/mnt/d/projects/02_academia/icml-repro"
CKPT_EXT = (".ckpt", ".pt", ".pth", ".h5", ".ckpt.index")


def find_ckpts(root):
    found = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(CKPT_EXT):
                found.append(os.path.join(dp, f))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper")
    ap.add_argument("dest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = os.path.join(SRC_ROOT, a.paper)
    if not os.path.isdir(src):
        raise SystemExit(f"no source folder: {src}")
    dest = os.path.abspath(a.dest)
    os.makedirs(dest, exist_ok=True)

    # 1) data/
    data_src = os.path.join(src, "data")
    data_dst = os.path.join(dest, "data")
    if os.path.isdir(data_src):
        if a.dry_run:
            n = sum(len(f) for _, _, f in os.walk(data_src))
            print(f"DRY: would copy data/ ({n} files) {data_src} -> {data_dst}")
        else:
            shutil.copytree(data_src, data_dst, dirs_exist_ok=True)
            print(f"copied data/ -> {data_dst}")
    else:
        print(f"(no data/ in {src})")

    # 2) checkpoints anywhere in paper folder
    ckpts = find_ckpts(src)
    if ckpts:
        ckdst = os.path.join(dest, "checkpoints")
        os.makedirs(ckdst, exist_ok=True)
        if a.dry_run:
            print(f"DRY: would copy {len(ckpts)} checkpoints -> {ckdst}")
        else:
            for c in ckpts:
                shutil.copy(c, os.path.join(ckdst, os.path.basename(c)))
            print(f"copied {len(ckpts)} checkpoints -> {ckdst}")
    else:
        print(f"(no checkpoints found in {src})")
    print("done")


if __name__ == "__main__":
    main()
