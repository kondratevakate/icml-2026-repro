# Data sourcing for cross-branch reproduction (codex/medical + others)

**Date:** 2026-08-02
**Context:** Pipeline arm2+compaction reruns all reproduced papers; many need datasets
+ checkpoints that are NOT in the git repo. They live on the Windows D: drive.

## Where the data is

- **Windows path:** `D:\projects\02_academia\icml-repro` (55 folders; ~48 are papers)
- **WSL mount:** `/mnt/d/projects/02_academia/icml-repro` (9P filesystem — SLOW, do not run agents here)
- Structure per paper folder `<paper>/`:
  - `official/code` — original paper code
  - `official/arxiv-source` — LaTeX source
  - `repro_*/` — someone's prior reproduction attempt
  - `data/` — **datasets + checkpoints** (e.g. `data/cifar100n/CIFAR-100_human.pt`)
  - `.trackio/` — already-published Space logbook

## Rule (DO NOT skip)

1. **Never run the agent against `/mnt/d/` directly** (9P is slow; WSL2 builds crash there).
2. Before rerunning a paper `<orid>`:
   ```
   mkdir -p <run_dir>/<orid>_arm2/data
   cp -r /mnt/d/projects/02_academia/icml-repro/<paper>/data/* <run_dir>/<orid>_arm2/data/
   # also copy any *.ckpt *.pt *.pth *.h5 checkpoints from official/ or repro_*/
   ```
3. Agent runs on **native ext4** (`/home/kate/...`), reads `./data/` locally.
4. After rerun → `local_to_trackio.py` + `publish_local.py` (compare-and-publish to HF Space).

## Which papers need data (from codex/medical branch DIRS)

`publish_logbooks.py` DIRS: globalhealthatlas, supgcl, glean, caml, dpsurv, lvcg,
bayes_causal_meta, sprout, medcrp_cl — all under `/mnt/d/.../icml-repro/<name>/data/`.

## Sync status (2026-08-02)

- The D: drive (55 folders, data/ + checkpoints) is **60 GB** and is being synced between
  machines. Until sync completes, `sync_data.py` works only on the machine that has D: mounted.
- `sync_data.py` copies per-paper `data/` + checkpoints to native FS before rerun — safe to
  run once D: is available.

## Publish-only-if-better rule (IMPORTANT)

`publish_local.py` SKIPs when `local_pts <= remote_pts`. So:
- A paper where a local arm scored LOWER than what's already in Space will NOT overwrite
  the better Space version. Example: CAffNet arm1 (6/10) vs Space arm2 (8/10) → SKIP,
  arm1's bundle is NOT published (arm2 already there and better).
- "Bundle 2" (the 10 CPU-friendly papers selected earlier, section 6e: TBSyYj4VV6,
  l35QweVxgn, tRsnpaRO0m, LJdacnMXkr, KqMqJpSMnQ, omkG80XURl, uiw8P2JGbW, ugjBMARbyt,
  rZTiFcDihH, vqxprtjuKH) only gets pushed if its score beats the live Space. Otherwise
  it stays local-only. These are the next batch after the pilot.
- To force-refresh a Space with identical score but improved logbook, use `--force`.

## TODO

- [ ] Build a `sync_data.py` helper: given `<orid>` + paper-name, copies data/ + checkpoints
      from D: into the run folder on native FS. (DONE)
- [ ] For each target paper, verify data/ exists on D: before launching agent.
- [ ] Note: D: is the SOURCE OF TRUTH for datasets; git repo stays code-only.
